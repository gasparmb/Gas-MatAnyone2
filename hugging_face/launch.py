import sys
import os
import threading
import socket
import time

# Must be set before importing custom_server (which imports torch)
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def wait_for_server(port, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


class MatAnyoneApi:
    """Python API exposed to JS via pywebview."""

    def __init__(self):
        self.window = None  # set after webview.create_window

    def download_result(self, filename, suggested_name=None):
        import shutil
        import webview
        src = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(src):
            return {'ok': False, 'error': 'Fichier introuvable'}
        save_name = suggested_name or filename
        if self.window:
            result = self.window.create_file_dialog(
                webview.SAVE_DIALOG,
                directory=os.path.expanduser('~/Downloads'),
                save_filename=save_name,
                file_types=('Vidéo MP4 (*.mp4)',),
            )
            if not result:
                return {'ok': False, 'cancelled': True}
            dst = result[0]
        else:
            dst = os.path.join(os.path.expanduser('~/Downloads'), save_name)
        shutil.copy2(src, dst)
        return {'ok': True, 'name': os.path.basename(dst)}

    def reveal_in_finder(self, filename):
        import subprocess
        src = os.path.join(RESULTS_DIR, filename)
        subprocess.run(['open', '-R', src])
        return {'ok': True}


if __name__ == '__main__':
    import webview

    port = find_free_port()

    # Import app after env vars are set
    from custom_server import app
    import uvicorn

    def run_server():
        uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print(f"Démarrage du serveur sur http://127.0.0.1:{port} …")
    if not wait_for_server(port):
        print("Le serveur n'a pas démarré dans les délais.")
        sys.exit(1)

    print("Ouverture de la fenêtre…")
    api = MatAnyoneApi()
    window = webview.create_window(  # noqa (api.window set below)
        title='MatAnyone 2',
        url=f'http://127.0.0.1:{port}',
        width=1440,
        height=900,
        resizable=True,
        min_size=(900, 600),
        js_api=api,
    )
    api.window = window
    webview.start(debug=False)
