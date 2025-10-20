"""
Real-time Tensor Monitoring System
Backend API using FastAPI
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import json
import psutil
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
import os

app = FastAPI(title="Tensolr Monitor", description="Real-time monitoring for tensor operations")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@dataclass
class TensorMetrics:
    """Data class for tensor metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_total: float
    gpu_usage: float  # Placeholder - would need actual GPU monitoring
    gpu_memory: float  # Placeholder - would need actual GPU monitoring
    tensor_operations: int
    active_tensors: int


class ConnectionManager:
    """Websocket connection manager for broadcasting metrics"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, data: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(data))
            except:
                # Remove disconnected client
                self.disconnect(connection)


manager = ConnectionManager()

# Global metrics storage
metrics_history: List[TensorMetrics] = []
current_metrics = TensorMetrics(
    timestamp=time.time(),
    cpu_usage=0.0,
    memory_usage=0.0,
    memory_total=0.0,
    gpu_usage=0.0,
    gpu_memory=0.0,
    tensor_operations=0,
    active_tensors=0
)

# Simulated tensor metrics tracker
tensor_operations_count = 0
active_tensors_count = 0


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(os.path.dirname(__file__), "index.html")) as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
            # We could implement commands from the client here if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/metrics/current")
async def get_current_metrics():
    """Get current tensor metrics"""
    global current_metrics
    return asdict(current_metrics)


@app.get("/api/metrics/history")
async def get_metrics_history(limit: int = 100):
    """Get historical tensor metrics"""
    global metrics_history
    # Return the last 'limit' metrics
    return [asdict(m) for m in metrics_history[-limit:]]


def update_metrics_loop():
    """Background loop to update metrics"""
    global current_metrics, metrics_history, tensor_operations_count, active_tensors_count
    
    while True:
        # Collect system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Update metrics
        current_metrics = TensorMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_percent,
            memory_usage=memory.used,
            memory_total=memory.total,
            gpu_usage=0.0,  # Would implement actual GPU monitoring
            gpu_memory=0.0,  # Would implement actual GPU monitoring
            tensor_operations=tensor_operations_count,
            active_tensors=active_tensors_count
        )
        
        # Store in history (keep last 1000 entries)
        metrics_history.append(current_metrics)
        if len(metrics_history) > 1000:
            metrics_history = metrics_history[-1000:]
        
        # Broadcast to all WebSocket clients
        asyncio.run(manager.broadcast(asdict(current_metrics)))
        
        # Update counts (in a real system, these would be updated by tensor operations)
        # For simulation, just increment periodically
        tensor_operations_count += 1
        if tensor_operations_count % 5 == 0:
            active_tensors_count = max(0, active_tensors_count + (1 if active_tensors_count < 20 else -1))
        
        time.sleep(1)  # Update every second


# Start the metrics collection in a background thread
metrics_thread = threading.Thread(target=update_metrics_loop, daemon=True)
metrics_thread.start()


# Additional endpoints for tensor-specific monitoring
@app.post("/api/tensor/register")
async def register_tensor(tensor_id: str, shape: List[int], dtype: str):
    """Register a new tensor for monitoring"""
    global active_tensors_count
    active_tensors_count += 1
    return {"message": f"Tensor {tensor_id} registered", "active_tensors": active_tensors_count}


@app.post("/api/tensor/remove")
async def remove_tensor(tensor_id: str):
    """Remove a tensor from monitoring"""
    global active_tensors_count
    active_tensors_count = max(0, active_tensors_count - 1)
    return {"message": f"Tensor {tensor_id} removed", "active_tensors": active_tensors_count}


@app.post("/api/tensor/operation")
async def tensor_operation(op_type: str, duration: float = 0.0):
    """Register a tensor operation"""
    global tensor_operations_count
    tensor_operations_count += 1
    return {"message": f"Operation {op_type} registered", "total_operations": tensor_operations_count}