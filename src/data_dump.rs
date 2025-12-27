use mirada_lib::database::Database;
use std::path::Path;

pub fn data_dump(database: String) {
    let database = Database::new(Path::new(&database));

    log::info!("Dumping out data from database...");
    database.dump();
}
