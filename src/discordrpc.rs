use discord_rpc_client::{models::rich_presence::Activity, Client};
use std::time::Instant;

pub struct DiscordRPC {
    rpc: Option<Client>,
    last_update: Instant,
    updatefreq: u64,
}

impl DiscordRPC {
    pub fn new(client_id: u64, updatefreq: u64) -> Self {
        let rpc = match Client::new(client_id) {
            Ok(mut rpc) => {
                rpc.start();

                Some(rpc)
            }
            Err(err) => {
                ::warn!("Failed to start DiscordRPC - {}", err);

                None
            }
        };

        let last_update = Instant::now();

        DiscordRPC {
            rpc,
            last_update,
            updatefreq,
        }
    }

    pub fn set_activity(&mut self, activity: impl FnOnce(Activity) -> Activity) {
        if let Some(ref mut rpc) = self.rpc {
            if self.last_update.elapsed().as_secs() > self.updatefreq {
                if let Err(err) = rpc.set_activity(activity) {
                    ::warn!("Failed to set activity - {}", err);
                }

                self.last_update = Instant::now();
            }
        }
    }
}

impl Drop for DiscordRPC {
    fn drop(&mut self) {
        if let Some(ref mut rpc) = self.rpc {
            if let Err(err) = rpc.clear_activity() {
                ::warn!("Failed to clear activity - {}", err);
            }
        }
    }
}
