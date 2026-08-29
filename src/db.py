import os
import datetime
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv()

# MongoDB Connection Configuration
DEFAULT_URI = "mongodb+srv://ameerhamza031946_db_user:E9sCrYIZu87CwmEq@cluster0.pqavtcq.mongodb.net/student_db?retryWrites=true&w=majority"
MONGODB_URI = os.getenv("MONGODB_URI", DEFAULT_URI)

_client = None
_db = None

def get_db_client():
    """Initializes and returns the PyMongo client and database object."""
    global _client, _db
    if _db is not None:
        return _db
    
    try:
        import pymongo
        uri = os.getenv("MONGODB_URI", DEFAULT_URI)
        # Avoid connecting if default placeholder username is unchanged
        if "<db_username>" in uri:
            return None
            
        _client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        # Verify connection
        _client.admin.command('ping')
        _db = _client.get_database("student_db")
        print("[SUCCESS] Successfully connected to MongoDB Atlas!")
        return _db
    except Exception as e:
        print(f"[NOTICE] MongoDB connection notice: {e}")
        return None

def check_db_status() -> Tuple[bool, str]:
    """Checks if MongoDB Atlas database connection is active."""
    try:
        db = get_db_client()
        if db is not None:
            return True, "Connected to MongoDB Atlas"
        else:
            uri = os.getenv("MONGODB_URI", DEFAULT_URI)
            if "<db_username>" in uri:
                return False, "Please update <db_username> in .env with your MongoDB Atlas username"
            return False, "Disconnected"
    except Exception as e:
        return False, str(e)

def save_single_prediction(student_input: Dict[str, Any], prediction_output: Dict[str, Any]) -> bool:
    """Saves individual student prediction result to MongoDB collection 'predictions'."""
    db = get_db_client()
    if db is None:
        return False
    try:
        doc = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "input_metrics": student_input,
            "prediction": prediction_output
        }
        db.predictions.insert_one(doc)
        return True
    except Exception as e:
        print(f"Failed to save prediction to MongoDB: {e}")
        return False

def save_batch_predictions(records: List[Dict[str, Any]]) -> bool:
    """Bulk saves batch student predictions to MongoDB collection 'batch_predictions'."""
    db = get_db_client()
    if db is None:
        return False
    try:
        docs = []
        now = datetime.datetime.now(datetime.timezone.utc)
        for rec in records:
            doc_rec = rec.copy()
            doc_rec["timestamp"] = now
            docs.append(doc_rec)
        if docs:
            db.batch_predictions.insert_many(docs)
        return True
    except Exception as e:
        print(f"Failed to save batch predictions to MongoDB: {e}")
        return False
