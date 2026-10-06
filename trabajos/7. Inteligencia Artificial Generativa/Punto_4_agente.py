from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# --- BASES DE DATOS SIMULADAS ---
CATALOGO = {"rodamiento": "REP-1001", "faja": "REP-2002", "sensor": "REP-3003"}
INVENTARIO = {
    "REP-1001": {"stock": 5, "ubicacion": "Estante A1"},
    "REP-2002": {"stock": 0, "ubicacion": "Estante B3"},
}
HISTORIAL = []
ORDENES = []

# --- DEFINICIÓN DE LAS 4 HERRAMIENTAS ---
@tool
def buscar_repuesto(descripcion: str) -> str:
    """Busca el código de un repuesto dado su nombre o descripción."""
    for nombre, codigo in CATALOGO.items():
        if nombre in descripcion.lower():
            return f"Código encontrado para '{descripcion}': {codigo}"
    return f"No se encontró código para el repuesto '{descripcion}'."

@tool
def consultar_almacen(codigo_repuesto: str) -> str:
    """Consulta el stock y ubicación de un repuesto mediante su código (ej: REP-1001)."""
    item = INVENTARIO.get(codigo_repuesto)
    if item:
        return f"Repuesto {codigo_repuesto}: Stock = {item['stock']} unidades, Ubicación = {item['ubicacion']}."
    return f"El código {codigo_repuesto} no existe en el almacén."

@tool
def generar_orden_trabajo(equipo: str, falla: str, codigo_repuesto: str) -> str:
    """Genera una orden de trabajo para reparar un equipo."""
    ot_id = f"OT-{len(ORDENES) + 1:04d}"
    ORDENES.append({"id": ot_id, "equipo": equipo, "falla": falla, "repuesto": codigo_repuesto})
    return f"Orden de Trabajo creada exitosamente con ID {ot_id} para el equipo {equipo}."

@tool
def actualizar_historial_fallas(equipo: str, falla: str, solucion: str) -> str:
    """Registra un evento en el historial de fallas del equipo."""
    HISTORIAL.append({"equipo": equipo, "falla": falla, "solucion": solucion})
    return f"Historial actualizado para el equipo {equipo} con la falla '{falla}'."

# --- CONFIGURACIÓN DEL AGENTE Y MODELO ---
tools = [buscar_repuesto, consultar_almacen, generar_orden_trabajo, actualizar_historial_fallas]
llm = ChatOllama(model="qwen2.5", temperature=0)
agent = create_react_agent(llm, tools)

# --- EJECUCIÓN ---
if __name__ == "__main__":
    prompt = (
        "El motor M-01 presentó una falla de sobrecalentamiento por un rodamiento dañado. "
        "Busca el repuesto, verifica si hay stock en el almacén, genera la orden de trabajo "
        "y actualiza el historial de fallas del equipo."
    )

    print("--- Procesando flujo del agente ---")
    resultado = agent.invoke({"messages": [("user", prompt)]})

    for msg in resultado["messages"]:
        print(f"\n[{msg.type.upper()}]: {msg.content}")
