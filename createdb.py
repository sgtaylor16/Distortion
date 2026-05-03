from dbsetup import Base, engine


def create_database() -> None:
	Base.metadata.create_all(engine)


if __name__ == "__main__":
	create_database()
