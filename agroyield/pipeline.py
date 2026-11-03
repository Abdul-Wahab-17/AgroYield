"""One command for the whole pipeline.

    python -m agroyield.pipeline all               # download -> merge -> train -> export
    python -m agroyield.pipeline download          # fetch every raw source (resumable)
    python -m agroyield.pipeline merge
    python -m agroyield.pipeline train [A|B]       # A = 1981+ (NASA POWER), B = all years
    python -m agroyield.pipeline export [A|B]      # build api/artifacts/

Every stage skips work that is already on disk, so an interrupted run resumes where it stopped.
Delete a file under data/ to redo that step. Rough cost on 8 cores: downloads 1-3 hours (mostly
NASA POWER and the 1.1 GB NASS file), merge a few minutes, training 2-4 hours per variant.
"""
import sys

STAGES = ("download", "merge", "train", "export")


def download():
    from . import download_climdiv, download_counties, download_nass, download_soil, download_weather
    download_counties.run()
    download_nass.run()
    download_climdiv.run()
    download_soil.run()
    download_weather.run()


def merge():
    from . import merge as m
    m.run()


def train(variant="A"):
    from . import train as t
    t.run(variant)


def export(variant="A"):
    from . import export as e
    e.run(variant)


def main(argv):
    if not argv or argv[0] not in STAGES + ("all",):
        sys.exit(__doc__)
    stage, rest = argv[0], argv[1:]
    for name in (STAGES if stage == "all" else (stage,)):
        print(f"\n===== {name} =====", flush=True)
        globals()[name](*rest) if name in ("train", "export") else globals()[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
