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
        print(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(data))
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected_clients.append(connection)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.active_connections:
                self.active_connections.remove(client)


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
            print(f"Received from WebSocket client: {data}")
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
        
        # Update the existing current_metrics object instead of creating a new one
        # This ensures that updates from API calls are preserved
        current_metrics.timestamp = time.time()
        current_metrics.cpu_usage = cpu_percent
        current_metrics.memory_usage = memory.used
        current_metrics.memory_total = memory.total
        current_metrics.gpu_usage = 0.0  # Would implement actual GPU monitoring
        current_metrics.gpu_memory = 0.0  # Would implement actual GPU monitoring
        # Note: tensor_operations_count and active_tensors_count should be updated via API calls
        # so we don't override them here
        
        # Store in history (keep last 1000 entries)
        metrics_history.append(current_metrics)
        if len(metrics_history) > 1000:
            metrics_history = metrics_history[-1000:]
        
        # Broadcast to all WebSocket clients
        asyncio.run(manager.broadcast(asdict(current_metrics)))
        
        time.sleep(1)  # Update every second


# Start the metrics collection in a background thread
metrics_thread = threading.Thread(target=update_metrics_loop, daemon=True)
metrics_thread.start()


# Additional endpoints for tensor-specific monitoring
# Define Pydantic models for request bodies
class TensorRegistration(BaseModel):
    tensor_id: str
    shape: List[int]
    dtype: str

class TensorRemoval(BaseModel):
    tensor_id: str

class TensorOperation(BaseModel):
    op_type: str
    duration: float = 0.0


def get_current_system_metrics():
    """Helper function to get up-to-date system metrics"""
    cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking call
    memory = psutil.virtual_memory()
    
    # For GPU, we'll use a placeholder since actual GPU monitoring might be slow
    gpu_percent = 0.0
    gpu_memory = 0.0
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            gpu_percent = gpu.load * 100
            gpu_memory = gpu.memoryUsed  # in MB
    except ImportError:
        pass
    except Exception:
        pass
    
    return {
        'cpu_usage': cpu_percent,
        'memory_usage': memory.used,
        'memory_total': memory.total,
        'gpu_usage': gpu_percent,
        'gpu_memory': gpu_memory
    }

@app.post("/api/tensor/register")
async def register_tensor(request: TensorRegistration):
    """Register a new tensor for monitoring"""
    global active_tensors_count
    active_tensors_count += 1
    # Get fresh system metrics and update the current metrics
    sys_metrics = get_current_system_metrics()
    current_metrics.active_tensors = active_tensors_count
    current_metrics.cpu_usage = sys_metrics['cpu_usage']
    current_metrics.memory_usage = sys_metrics['memory_usage']
    current_metrics.memory_total = sys_metrics['memory_total']
    current_metrics.gpu_usage = sys_metrics['gpu_usage']
    current_metrics.gpu_memory = sys_metrics['gpu_memory']
    current_metrics.timestamp = time.time()
    # Broadcast the updated metrics to all connected clients
    await manager.broadcast(asdict(current_metrics))
    return {"message": f"Tensor {request.tensor_id} registered", "active_tensors": active_tensors_count}


@app.post("/api/tensor/remove")
async def remove_tensor(request: TensorRemoval):
    """Remove a tensor from monitoring"""
    global active_tensors_count
    active_tensors_count = max(0, active_tensors_count - 1)
    # Get fresh system metrics and update the current metrics
    sys_metrics = get_current_system_metrics()
    current_metrics.active_tensors = active_tensors_count
    current_metrics.cpu_usage = sys_metrics['cpu_usage']
    current_metrics.memory_usage = sys_metrics['memory_usage']
    current_metrics.memory_total = sys_metrics['memory_total']
    current_metrics.gpu_usage = sys_metrics['gpu_usage']
    current_metrics.gpu_memory = sys_metrics['gpu_memory']
    current_metrics.timestamp = time.time()
    # Broadcast the updated metrics to all connected clients
    await manager.broadcast(asdict(current_metrics))
    return {"message": f"Tensor {request.tensor_id} removed", "active_tensors": active_tensors_count}


@app.post("/api/tensor/operation")
async def tensor_operation(request: TensorOperation):
    """Register a tensor operation"""
    global tensor_operations_count
    tensor_operations_count += 1
    # Get fresh system metrics and update the current metrics
    sys_metrics = get_current_system_metrics()
    current_metrics.tensor_operations = tensor_operations_count
    current_metrics.cpu_usage = sys_metrics['cpu_usage']
    current_metrics.memory_usage = sys_metrics['memory_usage']
    current_metrics.memory_total = sys_metrics['memory_total']
    current_metrics.gpu_usage = sys_metrics['gpu_usage']
    current_metrics.gpu_memory = sys_metrics['gpu_memory']
    current_metrics.timestamp = time.time()
    # Broadcast the updated metrics to all connected clients
    await manager.broadcast(asdict(current_metrics))
    return {"message": f"Operation {request.op_type} registered", "total_operations": tensor_operations_count}