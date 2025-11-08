import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, LargeBinary, DateTime
import numpy as np
import datetime

Base = declarative_base()

class Identity(Base):
    __tablename__ = 'identities'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    image_path = Column(String)
    embedding = Column(LargeBinary, nullable=False)
    added_at = Column(DateTime, default=datetime.datetime.utcnow)

class GalleryDB:
    def __init__(self, db_uri='sqlite:///data/gallery.db'):
        self.engine = sa.create_engine(db_uri, connect_args={'check_same_thread': False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add(self, name, image_path, emb):
        session = self.Session()
        obj = Identity(name=name, image_path=image_path, embedding=emb.astype('float32').tobytes())
        session.add(obj); session.commit(); session.refresh(obj)
        session.close()
        return obj.id

    def list(self):
        session = self.Session()
        rows = session.query(Identity).all()
        session.close()
        return [{'id': r.id, 'name': r.name, 'image_path': r.image_path, 'added_at': r.added_at.isoformat()} for r in rows]
