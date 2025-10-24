import sys
import json
import requests

BASE = "http://localhost:8080"

def pretty(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)

def create_game():
    r = requests.post(f"{BASE}/game/new", timeout=5)
    r.raise_for_status()
    return r.json()

def send_action(game_id, action):
    r = requests.post(f"{BASE}/game/{game_id}/action", json={"action": action}, timeout=5)
    r.raise_for_status()
    return r.json()

def map_user_action(user_action):
    a = user_action.strip().lower()
    if a in ("move", "forward"):
        return "Forward"
    if a in ("turn-left", "turnleft", "left"):
        return "TurnLeft"
    if a in ("turn-right", "turnright", "right"):
        return "TurnRight"
    if a in ("grab", "pickup", "pick"):
        return "Grab"
    if a in ("shoot", "fire"):
        return "Shoot"
    if a in ("climb", "exit"):
        return "Climb"
    return ""

def main():
    try:
        ngr = create_game()
    except requests.RequestException as e:
        print("Error creating game:", e, file=sys.stderr)
        sys.exit(1)

    game_id = ngr.get("gameId")
    print("Game ID:", game_id)
    print("Percepción inicial:")
    print(pretty(ngr.get("perception")))

    try:
        while True:
            user_in = input("Acción (move, turn-left, turn-right, grab, shoot, climb) o 'quit' para salir: ").strip()
            if not user_in:
                continue
            if user_in.lower() in ("quit", "exit"):
                print("Saliendo.")
                break

            server_action = map_user_action(user_in)
            if server_action == "":
                print("Acción desconocida. Usa move/turn-left/turn-right/grab/shoot/climb.")
                continue

            try:
                p = send_action(game_id, server_action)
            except requests.RequestException as e:
                print("Error al enviar acción:", e, file=sys.stderr)
                continue

            print("Percepción:")
            print(pretty(p))
            if isinstance(p, dict) and p.get("gameOver"):
                print("Juego terminado por el servidor.")
                break
    except (KeyboardInterrupt, EOFError):
        print("\nInterrumpido. Saliendo.")

if __name__ == "__main__":
    main()