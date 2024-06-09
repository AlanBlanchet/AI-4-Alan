from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

Base = declarative_base()
DATABASE_URI = "sqlite:///./back.db"

engine = create_engine(DATABASE_URI)
session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base.query = session.query_property()


class ItemQuery:
    def __init__(self, cls):
        self.cls = cls
        self._real = session.query(cls)

    def update(self, **kwargs):
        self._real.update(kwargs)
        return self._real.first()

    def delete(self):
        session.delete(self._real)
        session.commit()
        return True

    @classmethod
    def get(_, cls, id):
        query = ItemQuery(cls)
        query._real = query._real.filter_by(id=id)
        return query

    # def commit(self):
    #     session.commit()
    #     return self._real


class ManyQuery:
    def __init__(self, cls):
        self.cls = cls
        self._real = session.query(cls)

    def filter_by(self, **kwargs):
        self._real = self._real.filter_by(**kwargs)
        return self

    def filter(self, **kwargs):
        self._real = self._real.filter(**kwargs)
        return self

    def all(self) -> list:
        self._real = self._real.all()
        return self

    def update(self, **kwargs):
        self._real = self._real.update(kwargs)
        return self

    @classmethod
    def get(_, cls, id):
        query = ManyQuery(cls)
        query._real = query._real.get(id)
        return query

    def commit(self):
        session.commit()
        return self.cls(**self._real.__dict__)


class CustomBase(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True)

    def save(self):
        session.add(self)
        session.commit()
        return self

    def delete(self):
        session.delete(self)
        session.commit()
        return True

    @classmethod
    def filter(cls, **kwargs):
        return ManyQuery(cls).filter(**kwargs)

    @classmethod
    def filter_by(cls, **kwargs):
        return ManyQuery(cls).filter_by(**kwargs)

    @classmethod
    def get(cls, id) -> ManyQuery:
        return ItemQuery.get(cls, id)

    @classmethod
    def all(cls):
        return ManyQuery(cls).all()._real
