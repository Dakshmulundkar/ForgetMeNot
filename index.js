const { MongoClient } = require("mongodb");

const uri = "mongodb://localhost:27017";

const client = new MongoClient(uri);

async function run() {
  try {
    await client.connect();
    console.log("Connected to MongoDB!");

    // test insert
    const db = client.db("testdb");
    const collection = db.collection("users");

    await collection.insertOne({ name: "test user" });
    console.log("Inserted test data");
    
  } catch (err) {
    console.error(err);
  } finally {
    await client.close();
  }
}

run();
