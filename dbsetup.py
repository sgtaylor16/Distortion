from sqlalchemy import Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DB_NAME = "inlets.db"
DATABASE_URL = f"sqlite:///{DB_NAME}"

engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
	pass


class Inlets(Base):
	__tablename__ = "inlets"

	id: Mapped[int] = mapped_column(Integer, primary_key=True)
	source: Mapped[str] = mapped_column(String)
	desc: Mapped[str | None] = mapped_column(String)
	conditions: Mapped[list["Condition"]] = relationship(back_populates="inlet")

class Condition(Base):
	__tablename__ = "conditions"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True)
	inlet_id: Mapped[int] = mapped_column(ForeignKey("inlets.id"))
	altitude: Mapped[float] = mapped_column(Float)
	mach: Mapped[float] = mapped_column(Float)
	flow: Mapped[float] = mapped_column(Float)
	aoa: Mapped[float] = mapped_column(Float)
	aos: Mapped[float] = mapped_column(Float)
	inlet: Mapped["Inlets"] = relationship(back_populates="conditions")
	runs: Mapped[list["Runs"]] = relationship(back_populates="condition")

class Runs(Base):
	__tablename__ = "runs"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True)
	radius: Mapped[float] = mapped_column(Float, nullable=False)
	theta: Mapped[float] = mapped_column(Float, nullable=False)
	pt: Mapped[float] = mapped_column(Float)
	v_swirl: Mapped[float] = mapped_column(Float)
	v_tang: Mapped[float] = mapped_column(Float)
	v_axial: Mapped[float] = mapped_column(Float)
	mach: Mapped[float] = mapped_column(Float)
	condition_id: Mapped[int] = mapped_column(ForeignKey("conditions.id"))
	condition: Mapped["Condition"] = relationship(back_populates="runs")
