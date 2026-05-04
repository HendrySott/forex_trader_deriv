import logging


if __name__ == "__main__":
    try:
        from src.cli import main

        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
        print("Latest checkpoint has been saved in the 'training_checkpoints' folder")
        print("You can resume training from the main menu")
    except ModuleNotFoundError as e:
        print(f"\n❌ Missing Python dependency: {e.name}")
        print("Install the project dependencies, then run `python3 main.py` again.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.getLogger(__name__).error("Fatal error: %s", e, exc_info=True)
