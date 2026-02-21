import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/telemetry"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket telemetry!")
            await websocket.send(json.dumps({"action": "start"}))
            print("Sent start trigger...")
            
            for i in range(3):
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                print(f"Message {i+1}:")
                print(f"  Ep: {data.get('ep')}, Tick: {data.get('ep_tick')}")
                futures = data.get('futures', [])
                print(f"  Futures length: {len(futures)}")
                for h, f in enumerate(futures):
                    print(f"    [H={h}] x={f['x']:.3f}, y={f['y']:.3f}")
                print("-" * 20)
                
            print("Successfully verified Ghost payload stream!")
            await websocket.send(json.dumps({"action": "stop"}))
    except Exception as e:
        print(f"WebSocket test failed: {type(e).__name__}: {e}")

asyncio.run(test_websocket())
