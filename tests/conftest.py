from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db import Base


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as value:
        yield value


@pytest.fixture
def sample_video() -> Path:
    path = ROOT / "data/videos/sample_hardpoint.mp4"
    if not path.exists():
        pytest.skip("run `make fixture` first")
    return path

