use crate::data::{DataKey, StockData};
use heed::Database as Heed;
use heed::types::SerdeBincode;
use heed::{Env, EnvFlags, EnvOpenOptions, WithoutTls};
use std::path::Path;

pub const DATABASE_PATH: &str = "./database.lmdb";
pub const DATABASE_NAME: &str = "stocks";

#[derive(Clone)]
pub struct Database {
    database: Heed<SerdeBincode<DataKey>, SerdeBincode<StockData>>,
    env: Env<WithoutTls>,
}

impl Database {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();

        log::info!("Opening database environment in '{}'...", path.display());

        if !path.exists() {
            std::fs::create_dir_all(path).expect("Failed to create environment directory");
        }

        let env = unsafe {
            EnvOpenOptions::new()
                .read_txn_without_tls()
                .max_dbs(1)
                .max_readers(16)
                .flags(EnvFlags::empty())
                .open(path)
                .expect("Failed to open database")
        };

        let env_clone = env.clone();

        let database = {
            log::info!("Creating database '{DATABASE_NAME}'...");
            let mut write = env.write_txn().expect("Failed to get write lock");
            let db = env.create_database(&mut write, Some(DATABASE_NAME));
            write.commit().expect("Failed to commit database creation");
            db
        }
        .expect("Failed to open database");

        Self {
            database,
            env: env_clone,
        }
    }

    pub fn insert(&self, key: DataKey, data: StockData) {
        log::info!("Inserting data with {key}");
        let mut txn = self.env.write_txn().expect("Failed to get write lock");

        self.database
            .put(&mut txn, &key, &data)
            .expect("Failed to insert data");

        txn.commit().expect("Failed to commit data");
    }

    pub fn insert_many(&self, items: Vec<(DataKey, StockData)>) {
        let mut txn = self.env.write_txn().expect("Failed to get write lock");

        for (key, data) in items {
            self.database
                .put(&mut txn, &key, &data)
                .expect("Failed to insert data");
        }

        txn.commit().expect("Failed to commit data");
    }

    pub fn get(&self, key: DataKey) -> Option<StockData> {
        let ro = self.env.read_txn().expect("Failed to read lock database");

        self.database.get(&ro, &key).expect("Failed to get data")
    }

    pub fn dump(&self) {
        let read = self.env.read_txn().expect("Failed to read lock database");

        self.database.iter(&read).into_iter().for_each(|iter| {
            iter.enumerate().for_each(|(idx, item)| {
                let (key, _) = item.expect("Failed to get data item");

                println!("{idx} - {key}");
            })
        });
    }
}
