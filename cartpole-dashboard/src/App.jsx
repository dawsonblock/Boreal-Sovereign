import React, { useState, useEffect, useRef } from 'react';
import './index.css';

function App() {
  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [history, setHistory] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/telemetry');
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    ws.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      if (parsed.ep_tick === 0) {
        setHistory([]); // reset trailing history on new episode
      }
      setData(parsed);
      setHistory(prev => {
        const newHist = [...prev, parsed];
        if (newHist.length > 100) return newHist.slice(newHist.length - 100);
        return newHist;
      });
    };

    return () => {
      ws.close();
    };
  }, []);

  const handleStart = () => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({ action: 'start' }));
    }
  };

  const handleStop = () => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Boreal Apex Sovereign Engin​e — CartPole L3 Physics</h1>
        <div className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'Backend Online' : 'No Connection'}
        </div>
      </header>

      <div className="controls">
        <button className="btn btn-primary" onClick={handleStart} disabled={!isConnected}>
          Initialize CartPole Run
        </button>
        <button className="btn btn-danger" onClick={handleStop} disabled={!isConnected}>
          Halt Simulation
        </button>
        {data && <span className="ep-counter">Global Tick: {data.tick} | Episode: {data.ep} (Tick {data.ep_tick})</span>}
      </div>

      <div className="dashboard-grid">
        <div className="column-left">
          {/* Live Physical Simulation */}
          <div className="card animation-card">
            <h2>Live Physical Kinematics (L0 Syntax)</h2>
            <div className="cartpole-stage">
              <div className="track"></div>
              {/* Ghost Predictive Futures */}
              {data && data.futures && data.futures.map((future, idx) => (
                <div
                  key={`ghost-${idx}`}
                  className="cart ghost"
                  style={{
                    transform: `translateX(calc(${future.x * 120}px - 50%))`,
                    opacity: 0.15 + (1 - (idx / data.futures.length)) * 0.25, // Fades further out
                    zIndex: 5 - idx
                  }}
                >
                  <div className="pole" style={{ transform: `rotate(${future.y}rad)` }}></div>
                  <div className="cart-body"></div>
                </div>
              ))}

              {/* Actual Physical Cart */}
              {data && (
                <div
                  className="cart"
                  style={{
                    transform: `translateX(calc(${data.x * 120}px - 50%))`
                  }}
                >
                  <div className="pole" style={{ transform: `rotate(${data.y}rad)` }}></div>
                  <div className="cart-body"></div>
                  <div className="wheel wheel-left"></div>
                  <div className="wheel wheel-right"></div>
                </div>
              )}
            </div>
          </div>

          {/* Trajectory Map */}
          <div className="card map-card">
            <h2>1D Continuous State Space Mapping</h2>
            <div className="map-view cartpole-view">
              {history.map((pt, i) => (
                <div
                  key={i}
                  className="path-point"
                  style={{
                    // Map tick to X axis linearly, Map Pole angle to Y
                    left: `${(i / 100) * 100}%`,
                    bottom: `${((pt.y + 0.4) / 0.8) * 100}%`,
                    opacity: i / history.length,
                    backgroundColor: pt.gate_blocked ? '#ff2a2a' : '#00ffff'
                  }}
                />
              ))}
              {data && (
                <div
                  className="drone-marker"
                  style={{
                    left: `${(history.length / 100) * 100}%`,
                    bottom: `${((data.y + 0.4) / 0.8) * 100}%`,
                    backgroundColor: data.gate_blocked ? '#ff2a2a' : '#00ffff'
                  }}
                />
              )}
              {/* Safe boundaries for Cartpole (-0.209 to 0.209 radians) */}
              <div className="safe-boundary upper" style={{ bottom: `${((0.209 + 0.4) / 0.8) * 100}%` }}></div>
              <div className="safe-boundary lower" style={{ bottom: `${((-0.209 + 0.4) / 0.8) * 100}%` }}></div>
              <div className="target-marker" style={{ left: '50%', bottom: '50%' }}>★ Target Center</div>
            </div>
          </div>
        </div>

        {/* Cognitive Metrics */}
        <div className="card metrics-card">
          <h2>Cognitive Dynamics</h2>
          <div className="metrics-grid">
            <div className="metric">
              <span className="label">Pole Angle</span>
              <span className="value">{(data ? data.y : 0.0).toFixed(3)} rad</span>
            </div>
            <div className="metric">
              <span className="label">Cart Pos X</span>
              <span className="value">{(data ? data.x : 0.0).toFixed(3)}</span>
            </div>
            <div className="metric">
              <span className="label">Surprise (Free Energy)</span>
              <span className="value">{data ? data.surprise.toFixed(4) : '0.0000'}</span>
            </div>
            <div className="metric">
              <span className="label">Epistemic Uncertainty</span>
              <span className="value">{data ? data.uncertainty.toFixed(4) : '0.0000'}</span>
            </div>
            <div className="metric">
              <span className="label">L3 Regime Hash</span>
              <span className="value regime-hash">{data ? `0x${data.regime}` : 'INIT'}</span>
            </div>
            <div className="metric">
              <span className="label">Causal Mode</span>
              <span className={`value mode-${data?.mode?.toLowerCase()}`}>{data ? data.mode : 'READY'}</span>
            </div>
          </div>

          <h3>Sentinel Protective Gate</h3>
          <div className={`gate-status ${data?.gate_blocked ? 'blocked' : 'clear'}`}>
            {data?.gate_blocked ? '🛑 PREDICTIVE SHOCK ISOLATED (RECOVERY MODE)' : '✅ PHYSICS PREDICTION CLEAR'}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
