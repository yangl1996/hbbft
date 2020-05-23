use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::{fmt, result};

use byteorder::{BigEndian, ByteOrder};
use hex_fmt::{HexFmt, HexList};
use log::{debug, warn};
use rand::Rng;
use reed_solomon_erasure as rse;
use reed_solomon_erasure::{galois_8::Field as Field8, ReedSolomon};

use super::merkle::{Digest, MerkleTree, Proof};
use super::message::HexProof;
use super::{Error, FaultKind, Message, Result};
use crate::fault_log::Fault;
use crate::{ConsensusProtocol, NodeIdT, Target, ValidatorSet};

type RseResult<T> = result::Result<T, rse::Error>;

/// Broadcast algorithm instance.
#[derive(Debug)]
pub struct Broadcast<N> {
    /// Our ID.
    // TODO: Make optional for observers?
    our_id: N,
    our_share: Option<Proof<Vec<u8>>>,
    /// The set of validator IDs.
    val_set: Arc<ValidatorSet<N>>,
    /// The ID of the sending node.
    proposer_id: N,
    /// The Reed-Solomon erasure coding configuration.
    coding: Coding,
    /// If we are the proposer: whether we have already sent the `Value` messages with the shards.
    value_sent: bool,
    /// Whether we have already sent `Echo` to all nodes who haven't sent `CanDecode`.
    echo_sent: bool,
    /// Whether we have already multicast `Ready`.
    ready_sent: bool,
    /// Whether we have already output a value.
    decided: bool,
    /// Number of faulty nodes to optimize performance for.
    // TODO: Make this configurable: Allow numbers between 0 and N/3?
    fault_estimate: usize,
    /// The hashes we have received via `Echo` and `EchoHash` messages, by sender ID.
    echos: BTreeMap<N, EchoContent>,
    /// The root hashes we received via `Ready` messages, by sender ID.
    readys: BTreeMap<N, Vec<u8>>,
}

/// A `Broadcast` step, containing at most one output.
pub type Step<N> = crate::CpStep<Broadcast<N>>;

impl<N: NodeIdT> ConsensusProtocol for Broadcast<N> {
    type NodeId = N;
    type Input = Vec<u8>;
    type Output = Proof<Vec<u8>>;
    type Message = Message;
    type Error = Error;
    type FaultKind = FaultKind;

    fn handle_input<R: Rng>(&mut self, input: Self::Input, _rng: &mut R) -> Result<Step<N>> {
        self.broadcast(input)
    }

    fn handle_message<R: Rng>(
        &mut self,
        sender_id: &Self::NodeId,
        message: Message,
        _rng: &mut R,
    ) -> Result<Step<N>> {
        self.handle_message(sender_id, message)
    }

    fn terminated(&self) -> bool {
        self.decided
    }

    fn our_id(&self) -> &N {
        &self.our_id
    }
}

impl<N: NodeIdT> Broadcast<N> {
    /// Creates a new broadcast instance to be used by node `our_id` which expects a value proposal
    /// from node `proposer_id`.
    pub fn new<V>(our_id: N, val_set: V, proposer_id: N) -> Result<Self>
    where
        V: Into<Arc<ValidatorSet<N>>>,
    {
        let val_set: Arc<ValidatorSet<N>> = val_set.into();
        let parity_shard_num = 2 * val_set.num_faulty();
        let data_shard_num = val_set.num() - parity_shard_num;
        let coding =
            Coding::new(data_shard_num, parity_shard_num).map_err(|_| Error::InvalidNodeCount)?;
        let fault_estimate = val_set.num_faulty();

        Ok(Broadcast {
            our_id,
            our_share: None,
            val_set,
            proposer_id,
            coding,
            value_sent: false,
            echo_sent: false,
            ready_sent: false,
            decided: false,
            fault_estimate,
            echos: BTreeMap::new(),
            readys: BTreeMap::new(),
        })
    }

    /// Initiates the broadcast. This must only be called in the proposer node.
    pub fn broadcast(&mut self, input: Vec<u8>) -> Result<Step<N>> {
        if *self.our_id() != self.proposer_id {
            return Err(Error::InstanceCannotPropose);
        }
        if self.value_sent {
            return Err(Error::MultipleInputs);
        }
        self.value_sent = true;
        // Split the value into chunks/shards, encode them with erasure codes.
        // Assemble a Merkle tree from data and parity shards. Take all proofs
        // from this tree and send them, each to its own node.
        let (proof, step) = self.send_shards(input)?;
        let our_id = &self.our_id().clone();
        Ok(step.join(self.handle_value(our_id, proof)?))
    }

    /// Handles a message received from `sender_id`.
    ///
    /// This must be called with every message we receive from another node.
    pub fn handle_message(&mut self, sender_id: &N, message: Message) -> Result<Step<N>> {
        if !self.val_set.contains(sender_id) {
            return Err(Error::UnknownSender);
        }
        match message {
            Message::Value(p) => self.handle_value(sender_id, p),
            Message::Echo(h, p) => self.handle_echo(sender_id, &h, p),
            Message::Ready(ref hash) => self.handle_ready(sender_id, hash),
        }
    }

    /// Returns the proposer's node ID.
    pub fn proposer_id(&self) -> &N {
        &self.proposer_id
    }

    /// Returns the set of all validator IDs.
    pub fn validator_set(&self) -> &Arc<ValidatorSet<N>> {
        &self.val_set
    }

    /// Checks whether the conditions for output are met for this hash, and if so, sets the output
    /// value.
    fn compute_output(&mut self, hash: &Digest) -> Result<Step<N>> {
        // wait for 2n+1 Ready
        if self.decided
            || self.count_readys(hash) <= 2 * self.val_set.num_faulty()
        {
            return Ok(Step::default());
        }

        self.decided = true;
        Ok(Step::default().with_output(self.our_share.clone()))
    }

    /// Breaks the input value into shards of equal length and encodes them --
    /// and some extra parity shards -- with a Reed-Solomon erasure coding
    /// scheme. The returned value contains the shard assigned to this
    /// node. That shard doesn't need to be sent anywhere. It gets recorded in
    /// the broadcast instance.
    fn send_shards(&mut self, mut value: Vec<u8>) -> Result<(Proof<Vec<u8>>, Step<N>)> {
        let data_shard_num = self.coding.data_shard_count();
        let parity_shard_num = self.coding.parity_shard_count();

        // Insert the length of `v` so it can be decoded without the padding.
        let payload_len = value.len() as u32;
        value.splice(0..0, 0..4); // Insert four bytes at the beginning.
        BigEndian::write_u32(&mut value[..4], payload_len); // Write the size.
        let value_len = value.len(); // This is at least 4 now, due to the payload length.

        // Size of a Merkle tree leaf value: the value size divided by the number of data shards,
        // and rounded up, so that the full value always fits in the data shards. Always at least 1.
        let shard_len = (value_len + data_shard_num - 1) / data_shard_num;
        // Pad the last data shard with zeros. Fill the parity shards with zeros.
        value.resize(shard_len * (data_shard_num + parity_shard_num), 0);

        // Divide the vector into chunks/shards.
        let shards_iter = value.chunks_mut(shard_len);
        // Convert the iterator over slices into a vector of slices.
        let mut shards: Vec<&mut [u8]> = shards_iter.collect();

        // Construct the parity chunks/shards. This only fails if a shard is empty or the shards
        // have different sizes. Our shards all have size `shard_len`, which is at least 1.
        self.coding.encode(&mut shards).expect("wrong shard size");

        debug!(
            "{}: Value: {} bytes, {} per shard. Shards: {:0.10}",
            self,
            value_len,
            shard_len,
            HexList(&shards)
        );

        // Create a Merkle tree from the shards.
        let mtree = MerkleTree::from_vec(shards.into_iter().map(|shard| shard.to_vec()).collect());

        // Default result in case of `proof` error.
        let mut result = Err(Error::ProofConstructionFailed);
        assert_eq!(self.val_set.num(), mtree.values().len());

        let mut step = Step::default();
        // Send each proof to a node.
        for (id, index) in self.val_set.all_indices() {
            let proof = mtree.proof(*index).ok_or(Error::ProofConstructionFailed)?;
            if *id == *self.our_id() {
                // The proof is addressed to this node.
                result = Ok(proof);
            } else {
                // Rest of the proofs are sent to remote nodes.
                let msg = Target::node(id.clone()).message(Message::Value(proof));
                step.messages.push(msg);
            }
        }

        result.map(|proof| (proof, step))
    }

    /// Handles a received echo and verifies the proof it contains.
    fn handle_value(&mut self, sender_id: &N, p: Proof<Vec<u8>>) -> Result<Step<N>> {
        // If the sender is not the proposer or if this is not the first `Value`, ignore.
        if *sender_id != self.proposer_id {
            let fault_kind = FaultKind::ReceivedValueFromNonProposer;
            return Ok(Fault::new(sender_id.clone(), fault_kind).into());
        }

        match self.echos.get(self.our_id()) {
            // Multiple values from proposer.
            Some((hash, _)) if hash != p.root_hash() => {
                return Ok(Fault::new(sender_id.clone(), FaultKind::MultipleValues).into())
            }
            // Already received proof.
            Some((hash, _)) if hash == p.root_hash() => {
                warn!(
                    "Node {:?} received Value({:?}) multiple times from {:?}.",
                    self.our_id(),
                    HexProof(&p),
                    sender_id
                );
                return Ok(Step::default());
            }
            _ => (),
        };

        // If the proof is invalid, log the faulty node behavior and ignore.
        if !self.validate_proof(&p, &self.our_id()) {
            return Ok(Fault::new(sender_id.clone(), FaultKind::InvalidProof).into());
        }

        // Send the `Echo` message
        let echo_steps = self.send_echo(p.clone())?;
        self.our_share = Some(p);
        Ok(echo_steps)
    }

    /// Handles a received `Echo` message.
    fn handle_echo(&mut self, sender_id: &N, hash: &Digest, p: usize) -> Result<Step<N>> {
        // If the sender has already sent `Echo`, ignore.
        if let Some((old_hash, _)) = self.echos.get(sender_id) {
            if old_hash == hash {
                warn!(
                    "Node {:?} received Echo({}) multiple times from {:?}.",
                    self.our_id(),
                    &p,
                    sender_id,
                );
                return Ok(Step::default());
            } else {
                return Ok(Fault::new(sender_id.clone(), FaultKind::MultipleEchos).into());
            }
        }

        if let Some((h, _)) = self.echos.get(sender_id) {
            if h != hash {
                return Ok(Fault::new(sender_id.clone(), FaultKind::MultipleEchos).into());
            }
        }

        // If the proof is invalid, log the faulty-node behavior, and ignore.
        if !self.validate_index(p, sender_id) {
            return Ok(Fault::new(sender_id.clone(), FaultKind::InvalidProof).into());
        }

        // Save the proof for reconstructing the tree later.
        self.echos.insert(sender_id.clone(), ((*hash).to_vec(), p));

        let mut step = Step::default();

        // Upon receiving `N - f` `Echo`s with this root hash, multicast `Ready`.
        if !self.ready_sent && self.count_echos(&hash) >= self.val_set.num_correct() {
            step.extend(self.send_ready(&hash)?);
        }

        Ok(step)
    }

    /// Handles a received `Ready` message.
    fn handle_ready(&mut self, sender_id: &N, hash: &Digest) -> Result<Step<N>> {
        // If the sender has already sent a `Ready` before, ignore.
        if let Some(old_hash) = self.readys.get(sender_id) {
            if old_hash == hash {
                warn!(
                    "Node {:?} received Ready({:?}) multiple times from {:?}.",
                    self.our_id(),
                    hash,
                    sender_id
                );
                return Ok(Step::default());
            } else {
                return Ok(Fault::new(sender_id.clone(), FaultKind::MultipleReadys).into());
            }
        }

        self.readys.insert(sender_id.clone(), hash.to_vec());

        let mut step = Step::default();
        // Upon receiving f + 1 matching Ready(h) messages, if Ready
        // has not yet been sent, multicast Ready(h).
        if self.count_readys(hash) == self.val_set.num_faulty() + 1 && !self.ready_sent {
            // Enqueue a broadcast of a Ready message.
            step.extend(self.send_ready(hash)?);
        }

        if self.count_readys(hash) == 2 * self.val_set.num_faulty() + 1 && !self.ready_sent {
            // Enqueue a broadcast of a Ready message.
            step.extend(self.send_ready(hash)?);
        }

        Ok(step)
    }

    /// Sends `Echo` message to all left nodes and handles it.
    fn send_echo(&mut self, p: Proof<Vec<u8>>) -> Result<Step<N>> {
        if !self.val_set.contains(&self.our_id) {
            return Ok(Step::default());
        }
        let echo_msg = Message::Echo(*p.root_hash(), p.index());
        let mut step = Step::default();
        // Send `Echo` message to all non-validating nodes and the ones on our left.
        let msg = Target::all().message(echo_msg);
        step.messages.push(msg);
        let our_id = &self.our_id().clone();
        Ok(step.join(self.handle_echo(our_id, p.root_hash(), p.index())?))
    }

    /// Sends a `Ready` message and handles it. Does nothing if we are only an observer.
    fn send_ready(&mut self, hash: &Digest) -> Result<Step<N>> {
        self.ready_sent = true;
        if !self.val_set.contains(&self.our_id) {
            return Ok(Step::default());
        }
        let ready_msg = Message::Ready(*hash);
        let step: Step<_> = Target::all().message(ready_msg).into();
        let our_id = &self.our_id().clone();
        Ok(step.join(self.handle_ready(our_id, hash)?))
    }

    /// Returns `true` if the proof is valid and has the same index as the node ID.
    fn validate_proof(&self, p: &Proof<Vec<u8>>, id: &N) -> bool {
        self.val_set.index(id) == Some(p.index()) && p.validate(self.val_set.num())
    }

    /// Returns `true` if the given value is the same as the index of the node.
    fn validate_index(&self, p: usize, id: &N) -> bool {
        self.val_set.index(id) == Some(p)
    }

    /// Returns the number of nodes that have sent us an `Echo` or `EchoHash` message with this hash.
    fn count_echos(&self, hash: &Digest) -> usize {
        self.echos.values().filter(|v| v.0 == hash).count()
    }

    /// Returns the number of nodes that have sent us a `Ready` message with this hash.
    fn count_readys(&self, hash: &Digest) -> usize {
        self.readys
            .values()
            .filter(|h| h.as_slice() == hash)
            .count()
    }
}

impl<N: NodeIdT> fmt::Display for Broadcast<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> result::Result<(), fmt::Error> {
        write!(f, "{:?} Broadcast({:?})", self.our_id(), self.proposer_id)
    }
}

/// A wrapper for `ReedSolomon` that doesn't panic if there are no parity shards.
#[derive(Debug)]
enum Coding {
    /// A `ReedSolomon` instance with at least one parity shard.
    ReedSolomon(Box<ReedSolomon<Field8>>),
    /// A no-op replacement that doesn't encode or decode anything.
    Trivial(usize),
}

impl Coding {
    /// Creates a new `Coding` instance with the given number of shards.
    fn new(data_shard_num: usize, parity_shard_num: usize) -> RseResult<Self> {
        Ok(if parity_shard_num > 0 {
            let rs = ReedSolomon::new(data_shard_num, parity_shard_num)?;
            Coding::ReedSolomon(Box::new(rs))
        } else {
            Coding::Trivial(data_shard_num)
        })
    }

    /// Returns the number of data shards.
    fn data_shard_count(&self) -> usize {
        match *self {
            Coding::ReedSolomon(ref rs) => rs.data_shard_count(),
            Coding::Trivial(dsc) => dsc,
        }
    }

    /// Returns the number of parity shards.
    fn parity_shard_count(&self) -> usize {
        match *self {
            Coding::ReedSolomon(ref rs) => rs.parity_shard_count(),
            Coding::Trivial(_) => 0,
        }
    }

    /// Constructs (and overwrites) the parity shards.
    fn encode(&self, slices: &mut [&mut [u8]]) -> RseResult<()> {
        match *self {
            Coding::ReedSolomon(ref rs) => rs.encode(slices),
            Coding::Trivial(_) => Ok(()),
        }
    }

    /// If enough shards are present, reconstructs the missing ones.
    fn reconstruct_shards(&self, shards: &mut [Option<Box<[u8]>>]) -> RseResult<()> {
        match *self {
            Coding::ReedSolomon(ref rs) => rs.reconstruct(shards),
            Coding::Trivial(_) => {
                if shards.iter().all(Option::is_some) {
                    Ok(())
                } else {
                    Err(rse::Error::TooFewShardsPresent)
                }
            }
        }
    }
}

type EchoContent = (Vec<u8>, usize);

