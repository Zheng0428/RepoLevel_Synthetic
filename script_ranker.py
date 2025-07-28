import json
import utils




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract bugs and evaluate GPT bug detection with retry mechanism.')

    parser.add_argument('--max_quantity', type=int, default=5, 
                        help='Maximum number of retries for failed items (default: 5). Set to 1 to disable retries.')
    parser.add_argument('--structure_path', action='store_true', 
                        help='Enable the retry mechanism for failed items. If not set, runs only once.')

    args = parser.parse_args()

    