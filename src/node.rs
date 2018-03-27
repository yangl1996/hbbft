//! Networking controls of the consensus node.
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::{Send, Sync};
use std::net::{TcpListener, SocketAddr};
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use spmc;
use crossbeam;

use broadcast;
use proto::Message;
use commst;

/// This is a structure to start a consensus node.
pub struct Node<T> {
    /// Incoming connection socket.
    addr: SocketAddr,
    /// Sockets of remote nodes. TODO.
    remotes: Vec<SocketAddr>,
    /// Optionally, a value to be broadcast by this node.
    value: Option<T>
}

impl<T: Clone + Debug + Eq + Hash + Send + Sync + From<Vec<u8>> + AsRef<[u8]>>
    Node<T>
where Vec<u8>: From<T>
{
    /// Consensus node constructor. It only initialises initial parameters.
    pub fn new(addr: SocketAddr, remotes: Vec<SocketAddr>, value: Option<T>) ->
        Self
    {
        Node {addr, remotes, value}
    }

    /// Consensus node procedure implementing HoneyBadgerBFT.
    pub fn run(&self) -> Result<T, ()>
    {
        // Listen for incoming connections on a given TCP port.
        let listener = TcpListener::bind(&self.addr).unwrap();
        // Multicast channel from the manager task to comms tasks.
        let (stx, srx): (spmc::Sender<Message<T>>,
                         spmc::Receiver<Message<T>>) = spmc::channel();
        // Unicast channel from comms tasks to the manager task.
        let (mtx, mrx): (mpsc::Sender<Message<T>>,
                         mpsc::Receiver<Message<T>>) = mpsc::channel();
        let broadcast_value = self.value.to_owned();

        // All spawned threads will have exited by the end of the scope.
        crossbeam::scope(|scope| {

            // Listen for incoming socket connections and start a comms task for
            // each new connection.
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        info!("New connection from {:?}",
                              stream.peer_addr().unwrap());
                        let tx = mtx.clone();
                        let rx = srx.clone();
                        scope.spawn(move || {
                            commst::CommsTask::new(tx, rx, stream).run();
                        });

                        // TODO: break when all the remotes have joined
                    }
                    Err(e) => {
                        warn!("Failed to connect: {}", e);
                    }
                }
            }

            // broadcast stage
            let (tx, rx) = (Arc::new(Mutex::new(stx)),
                            Arc::new(Mutex::new(mrx)));
            match broadcast::Stage::new(tx, rx, broadcast_value).run() {
                Ok(_) => debug!("Broadcast stage succeeded"),
                Err(_) => error!("Broadcast stage failed")
            }

            // TODO: other stages

        }); // end of thread scope

        Err(())
    }
}
