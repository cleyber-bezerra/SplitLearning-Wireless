import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Script para gerenciar servidores.")
    parser.add_argument(
        'mode', 
        choices=['sync', 'async'], 
        help="Escolha o modo do servidor: 'sync' para servidor síncrono, 'async' para servidor assíncrono"
    )

    args = parser.parse_args()

    if args.mode == 'sync':
        server_module = importlib.import_module('servers.server_sync')
    elif args.mode == 'async':
        server_module = importlib.import_module('servers.server_async')

    server_module.main()

if __name__ == "__main__":
    main()
