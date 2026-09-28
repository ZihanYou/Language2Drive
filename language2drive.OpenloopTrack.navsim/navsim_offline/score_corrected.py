"""Score with geometric contract support enabled and required."""
import argparse


def main():
    try:
        import shapely  # noqa: F401
    except ImportError:
        argparse.ArgumentParser().error(
            "Shapely>=2 is required for corrected geometric scoring."
        )
    from score import main as score_main
    score_main()


if __name__ == "__main__":
    main()
