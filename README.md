# TFM: Gestión Dinámica del Riesgo y Rebalanceo de Carteras (Renta Variable)
Borrar credenciales Github 

2. Desde la terminal (PowerShell)
powershell
# Eliminar credenciales de GitHub específicamente
git credential reject
O de forma más directa:

powershell
# Listar y borrar con cmdkey
cmdkey /delete:LegacyGeneric:target=git:https://github.com
3. Git credential helper — borrar todo el caché
powershell
# Si usas el gestor de credenciales de Git
git credential-manager delete https://github.com

Este proyecto implementa un sistema en Python para la gestión dinámica del riesgo y el rebalanceo de carteras de inversión. Utiliza datos intradiarios de Tiingo, datos macroeconómicos de FRED, XGBoost como filtro de activos, y un Agente Autónomo para el control de la cartera.

## Requisitos

1. Crear un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configurar variables de entorno:
   Copiar `.env.example` a `.env` y configurar las API keys.

## Ejecución

Para ejecutar el pipeline principal:

```bash
python main_pipeline.py
```