"""Run with: python -m app.seed
Populates a few sample HCPs so the chat/form demo has something to search for.
"""
from app.database import SessionLocal, Base, engine
from app.models import HCP

Base.metadata.create_all(bind=engine)

SAMPLE_HCPS = [
    dict(name="Dr. Anjali Mehta", specialty="Cardiologist", hospital="Fortis Hospital",
         city="Mumbai", email="anjali.mehta@example.com", phone="+91-9800000001", tier="A"),
    dict(name="Dr. Rohan Kulkarni", specialty="Endocrinologist", hospital="Apollo Hospital",
         city="Pune", email="rohan.kulkarni@example.com", phone="+91-9800000002", tier="B"),
    dict(name="Dr. Sara Iyer", specialty="Oncologist", hospital="Tata Memorial",
         city="Mumbai", email="sara.iyer@example.com", phone="+91-9800000003", tier="A"),
    dict(name="Dr. Vikram Desai", specialty="General Physician", hospital="Ruby Hall Clinic",
         city="Pune", email="vikram.desai@example.com", phone="+91-9800000004", tier="C"),
]


def seed():
    db = SessionLocal()
    try:
        for data in SAMPLE_HCPS:
            existing = db.query(HCP).filter(HCP.name == data["name"]).first()
            if not existing:
                db.add(HCP(**data))
        db.commit()
        print(f"Seeded {len(SAMPLE_HCPS)} HCPs (skipping any that already exist).")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
