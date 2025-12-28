use mirada_lib::database::Database;
use std::path::Path;

pub fn dump(database: String, kind: String) {
    match kind.as_str() {
        "database" => {
            let database = Database::new(Path::new(&database));

            log::info!("Dumping out data from database...");
            database.dump();
        }
        _ => panic!("Invalid data kind. Available: 'database'."),
    }
}
