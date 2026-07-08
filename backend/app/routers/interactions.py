from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas

router = APIRouter()


@router.post("/hcps", response_model=schemas.HCPOut)
def create_hcp(hcp: schemas.HCPCreate, db: Session = Depends(get_db)):
    db_hcp = models.HCP(**hcp.model_dump())
    db.add(db_hcp)
    db.commit()
    db.refresh(db_hcp)
    return db_hcp


@router.get("/hcps", response_model=list[schemas.HCPOut])
def list_hcps(q: str | None = None, db: Session = Depends(get_db)):
    query = db.query(models.HCP)
    if q:
        like = f"%{q}%"
        query = query.filter(models.HCP.name.ilike(like))
    return query.order_by(models.HCP.name).all()


@router.get("/hcps/{hcp_id}", response_model=schemas.HCPOut)
def get_hcp(hcp_id: str, db: Session = Depends(get_db)):
    hcp = db.query(models.HCP).filter(models.HCP.id == hcp_id).first()
    if not hcp:
        raise HTTPException(404, "HCP not found")
    return hcp


@router.post("/interactions", response_model=schemas.InteractionOut)
def create_interaction(payload: schemas.InteractionCreate, db: Session = Depends(get_db)):
    """Direct-form submission path (bypasses the agent) — still available for
    reps who prefer the structured form end-to-end without chatting."""
    hcp = db.query(models.HCP).filter(models.HCP.id == payload.hcp_id).first()
    if not hcp:
        raise HTTPException(404, "HCP not found")
    data = payload.model_dump()
    interaction = models.Interaction(**data)
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


@router.get("/interactions", response_model=list[schemas.InteractionOut])
def list_interactions(hcp_id: str | None = None, db: Session = Depends(get_db)):
    query = db.query(models.Interaction)
    if hcp_id:
        query = query.filter(models.Interaction.hcp_id == hcp_id)
    return query.order_by(models.Interaction.interaction_date.desc()).all()


@router.get("/interactions/{interaction_id}", response_model=schemas.InteractionOut)
def get_interaction(interaction_id: str, db: Session = Depends(get_db)):
    interaction = db.query(models.Interaction).filter(models.Interaction.id == interaction_id).first()
    if not interaction:
        raise HTTPException(404, "Interaction not found")
    return interaction


@router.patch("/interactions/{interaction_id}", response_model=schemas.InteractionOut)
def update_interaction(interaction_id: str, payload: schemas.InteractionUpdate, db: Session = Depends(get_db)):
    """Direct-form edit path (bypasses the agent)."""
    interaction = db.query(models.Interaction).filter(models.Interaction.id == interaction_id).first()
    if not interaction:
        raise HTTPException(404, "Interaction not found")
    updates = payload.model_dump(exclude_unset=True)
    for k, v in updates.items():
        setattr(interaction, k, v)
    interaction.is_edited = True
    db.commit()
    db.refresh(interaction)
    return interaction
