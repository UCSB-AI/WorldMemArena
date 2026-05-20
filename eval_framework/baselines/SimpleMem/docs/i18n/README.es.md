<div align="center">

<img alt="simplemem_logo" src="https://github.com/user-attachments/assets/6ea54ad1-e007-442c-99d7-1174b10d1fec" width="450">

<div align="center">

## Memoria Eficiente de por Vida para Agentes LLM — Text & Multimodal

<small>Almacena, comprime y recupera memorias a largo plazo con compresión semántica sin pérdidas. Compatible con Claude, Cursor, LM Studio y más.</small>

</div>

<p><b>Funciona con cualquier plataforma de IA que soporte MCP o integración con Python</b></p>

<table>
<tr>

<td align="center" width="100">
  <a href="https://www.anthropic.com/claude">
    <img src="https://cdn.simpleicons.org/claude/D97757" width="48" height="48" alt="Claude Desktop" />
  </a><br/>
  <sub>
    <a href="https://www.anthropic.com/claude"><b>Claude Desktop</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://cursor.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/cursor/FFFFFF">
      <img src="https://cdn.simpleicons.org/cursor/000000" width="48" height="48" alt="Cursor" />
    </picture>
  </a><br/>
  <sub>
    <a href="https://cursor.com"><b>Cursor</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://lmstudio.ai">
    <img src="https://github.com/lmstudio-ai.png?size=200" width="48" height="48" alt="LM Studio" />
  </a><br/>
  <sub>
    <a href="https://lmstudio.ai"><b>LM Studio</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://cherry-ai.com">
    <img src="https://github.com/CherryHQ.png?size=200" width="48" height="48" alt="Cherry Studio" />
  </a><br/>
  <sub>
    <a href="https://cherry-ai.com"><b>Cherry Studio</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://pypi.org/project/simplemem/">
    <img src="https://cdn.simpleicons.org/pypi/3775A9" width="48" height="48" alt="PyPI" />
  </a><br/>
  <sub>
    <a href="https://pypi.org/project/simplemem/"><b>Paquete PyPI</b></a>
  </sub>
</td>

<td align="center" width="100">
  <sub><b>+ Cualquier<br/>Cliente MCP</b></sub>
</td>

</tr>
</table>

<div align="center">

<br/>

[🇨🇳 中文](./README.zh-CN.md) •
[🇯🇵 日本語](./README.ja.md) •
[🇰🇷 한국어](./README.ko.md) •
[🇪🇸 **Español**](./README.es.md) •
[🇫🇷 Français](./README.fr.md) •
[🇩🇪 Deutsch](./README.de.md) •
[🇧🇷 Português](./README.pt-br.md)<br/>
[🇷🇺 Русский](./README.ru.md) •
[🇸🇦 العربية](./README.ar.md) •
[🇮🇹 Italiano](./README.it.md) •
[🇻🇳 Tiếng Việt](./README.vi.md) •
[🇹🇷 Türkçe](./README.tr.md)

<br/>

[![Project Page](https://img.shields.io/badge/🎬_DEMO_INTERACTIVO-Visitar_Sitio_Web-FF6B6B?style=for-the-badge&labelColor=FF6B6B&color=4ECDC4&logoColor=white)](https://aiming-lab.github.io/SimpleMem-Page)

<p align="center">
  <a href="https://arxiv.org/abs/2601.02553"><img src="https://img.shields.io/badge/arXiv-2601.02553-b31b1b?style=flat&labelColor=555" alt="arXiv"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/github-SimpleMem-181717?style=flat&labelColor=555&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/github/license/aiming-lab/SimpleMem?style=flat&label=license&labelColor=555&color=2EA44F" alt="License"></a>
  <a href="https://github.com/aiming-lab/SimpleMem/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat&labelColor=555" alt="PRs Welcome"></a>
  <br/>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/v/simplemem?style=flat&label=pypi&labelColor=555&color=3775A9&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/pyversions/simplemem?style=flat&label=python&labelColor=555&color=3775A9&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://mcp.simplemem.cloud"><img src="https://img.shields.io/badge/MCP-mcp.simplemem.cloud-14B8A6?style=flat&labelColor=555" alt="MCP Server"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/Claude_Skills-supported-FFB000?style=flat&labelColor=555" alt="Claude Skills"></a>
  <br/>
  <a href="https://discord.gg/KA2zC32M"><img src="https://img.shields.io/badge/Discord-Unirse_al_Chat-5865F2?style=flat&labelColor=555&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="../../fig/wechat_logo3.JPG"><img src="https://img.shields.io/badge/WeChat-Grupo-07C160?style=flat&labelColor=555&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

<br/>

[Descripción General](#-descripción-general) • [Inicio Rápido](#-inicio-rápido) • [Servidor MCP](#-servidor-mcp) • [Evaluación](#-evaluación) • [Citación](#-citación)

</div>

</div>

<br/>

## 🔥 Novedades

- **[02/09/2026]** 🚀 **¡Memoria Cross-Session disponible - Superando a Claude-Mem en 64%!** SimpleMem ahora soporta **memoria persistente entre conversaciones**. En el benchmark LoCoMo, SimpleMem logra una **mejora del 64%** sobre Claude-Mem. Tus agentes ahora pueden recordar automáticamente el contexto, decisiones y aprendizajes de sesiones anteriores. [Ver Documentación Cross-Session →](../../cross/README.md)
- **[01/20/2026]** **¡SimpleMem ya está disponible en PyPI!** 📦 Instala directamente con `pip install simplemem`. [Ver Guía de Uso del Paquete →](../PACKAGE_USAGE.md)
- **[01/19/2026]** **¡Almacenamiento de memoria local añadido a SimpleMem Skill!** 💾 Ahora soporta almacenamiento de memoria local dentro de Claude Skills.
- **[01/18/2026]** **¡SimpleMem ahora soporta Claude Skills!** 🚀 Usa SimpleMem en claude.ai para memoria a largo plazo entre conversaciones. Regístrate en [mcp.simplemem.cloud](https://mcp.simplemem.cloud), configura tu token e importa el skill.
- **[01/14/2026]** **¡El servidor MCP de SimpleMem está EN VIVO y es Open Source!** 🎉 Servicio de memoria en la nube en [mcp.simplemem.cloud](https://mcp.simplemem.cloud). Se integra con LM Studio, Cherry Studio, Cursor, Claude Desktop mediante el protocolo MCP **Streamable HTTP**. [Ver Documentación MCP →](../../MCP/README.md)
- **[01/08/2026]** 🔥 ¡Únete a nuestro [Discord](https://discord.gg/KA2zC32M) y [grupo de WeChat](../../fig/wechat_logo3.JPG) para colaborar e intercambiar ideas!
- **[01/05/2026]** ¡El paper de SimpleMem fue publicado en [arXiv](https://arxiv.org/abs/2601.02553)!

---

## 📑 Tabla de Contenidos

- [🌟 Descripción General](#-descripción-general)
- [🎯 Contribuciones Principales](#-contribuciones-principales)
- [🚀 Aspectos Destacados del Rendimiento](#-aspectos-destacados-del-rendimiento)
- [📦 Instalación](#-instalación)
- [⚡ Inicio Rápido](#-inicio-rápido)
- [🔌 Servidor MCP](#-servidor-mcp)
- [📊 Evaluación](#-evaluación)
- [📝 Citación](#-citación)
- [📄 Licencia](#-licencia)
- [🙏 Agradecimientos](#-agradecimientos)

---

## 🌟 Descripción General

<div align="center">
<img src="../../fig/Fig_tradeoff.png" alt="Compromiso Rendimiento vs Eficiencia" width="900"/>

*SimpleMem logra una puntuación F1 superior (43.24%) con un costo mínimo de tokens (~550), ocupando la posición ideal superior izquierda.*
</div>

**SimpleMem** es un marco de memoria eficiente basado en **compresión semántica sin pérdidas** que aborda el desafío fundamental de la **memoria eficiente a largo plazo para agentes LLM**. A diferencia de los sistemas existentes que acumulan pasivamente contexto redundante o dependen de costosos bucles de razonamiento iterativo, SimpleMem maximiza la **densidad de información** y la **utilización de tokens** a través de un pipeline de tres etapas:

<table>
<tr>
<td width="33%" align="center">

### 🔍 Etapa 1
**Compresión Estructurada Semántica**

Destila interacciones no estructuradas en unidades de memoria compactas con índices multi-vista

</td>
<td width="33%" align="center">

### 🗂️ Etapa 2
**Síntesis Semántica en Línea**

Proceso intra-sesión que integra instantáneamente contexto relacionado en representaciones abstractas unificadas para eliminar redundancia

</td>
<td width="33%" align="center">

### 🎯 Etapa 3
**Planificación de Recuperación Consciente de la Intención**

Infiere la intención de búsqueda para determinar dinámicamente el alcance de recuperación y construir contexto preciso eficientemente

</td>
</tr>
</table>

<div align="center">
<img src="../../fig/Fig_framework.png" alt="Marco SimpleMem" width="900"/>

*La Arquitectura SimpleMem: (1) La Compresión Estructurada Semántica filtra diálogos de baja utilidad y convierte ventanas informativas en unidades de memoria compactas e independientes del contexto. (2) La Síntesis Semántica en Línea consolida fragmentos relacionados durante la escritura, manteniendo una topología de memoria compacta y coherente. (3) La Planificación de Recuperación Consciente de la Intención infiere la intención de búsqueda para adaptar el alcance de recuperación y las formas de consulta, permitiendo recuperación paralela multi-vista y construcción de contexto eficiente en tokens.*
</div>

---

### 🏆 Comparación de Rendimiento

<div align="center">

**Demo de Comparación de Velocidad**

<video src="https://github.com/aiming-lab/SimpleMem/raw/main/fig/simplemem-new.mp4" controls width="900"></video>

*SimpleMem vs. Línea Base: Demostración de comparación de velocidad en tiempo real*

</div>

<div align="center">

**Resultados del Benchmark LoCoMo-10 (GPT-4.1-mini)**

| Modelo | ⏱️ Tiempo de Construcción | 🔎 Tiempo de Recuperación | ⚡ Tiempo Total | 🎯 F1 Promedio |
|:------|:--------------------:|:-----------------:|:-------------:|:-------------:|
| A-Mem | 5140.5s | 796.7s | 5937.2s | 32.58% |
| LightMem | 97.8s | 577.1s | 675.9s | 24.63% |
| Mem0 | 1350.9s | 583.4s | 1934.3s | 34.20% |
| **SimpleMem** ⭐ | **92.6s** | **388.3s** | **480.9s** | **43.24%** |

</div>

> **💡 Ventajas Clave:**
> - 🏆 **Mayor Puntuación F1**: 43.24% (+26.4% vs. Mem0, +75.6% vs. LightMem)
> - ⚡ **Recuperación Más Rápida**: 388.3s (32.7% más rápido que LightMem, 51.3% más rápido que Mem0)
> - 🚀 **Más Rápido de Extremo a Extremo**: 480.9s de tiempo total de procesamiento (12.5× más rápido que A-Mem)

---

## 🎯 Contribuciones Principales

### 1️⃣ Compresión Estructurada Semántica

SimpleMem aplica un mecanismo de **filtrado implícito de densidad semántica** integrado en el proceso de generación del LLM para filtrar contenido de interacción redundante. El sistema reformula flujos de diálogo brutos en **unidades de memoria compactas** — hechos autocontenidos con correferencias resueltas y marcas de tiempo absolutas. Cada unidad se indexa mediante tres representaciones complementarias:

<div align="center">

| 🔍 Capa | 📊 Tipo | 🎯 Propósito | 🛠️ Implementación |
|---------|---------|------------|-------------------|
| **Semántica** | Densa | Similitud conceptual | Embeddings vectoriales (1024-d) |
| **Léxica** | Dispersa | Coincidencia exacta de términos | Índice de palabras clave estilo BM25 |
| **Simbólica** | Metadatos | Filtrado estructurado | Marcas de tiempo, entidades, personas |

</div>

**✨ Ejemplo de Transformación:**
```diff
- Entrada:  "Él se reunirá con Bob mañana a las 2pm"  [❌ relativo, ambiguo]
+ Salida: "Alice se reunirá con Bob en Starbucks el 2025-11-16T14:00:00"  [✅ absoluto, atómico]
```

---

### 2️⃣ Síntesis Semántica en Línea

A diferencia de los sistemas tradicionales que dependen del mantenimiento asíncrono en segundo plano, SimpleMem realiza la síntesis **sobre la marcha durante la fase de escritura**. Las unidades de memoria relacionadas se sintetizan en representaciones abstractas de nivel superior dentro del alcance de la sesión actual, permitiendo que experiencias repetitivas o estructuralmente similares se **desruiden y compriman inmediatamente**.

**✨ Ejemplo de Síntesis:**
```diff
- Fragmento 1: "El usuario quiere café"
- Fragmento 2: "El usuario prefiere leche de avena"
- Fragmento 3: "El usuario lo quiere caliente"
+ Consolidado: "El usuario prefiere café caliente con leche de avena"
```

Esta síntesis proactiva asegura que la topología de memoria se mantenga compacta y libre de fragmentación redundante.

---

### 3️⃣ Planificación de Recuperación Consciente de la Intención

En lugar de una recuperación de profundidad fija, SimpleMem aprovecha las capacidades de razonamiento del LLM para generar un **plan de recuperación integral**. Dada una consulta, el módulo de planificación infiere la **intención de búsqueda latente** para determinar dinámicamente el alcance y la profundidad de recuperación:

$$\{ q_{\text{sem}}, q_{\text{lex}}, q_{\text{sym}}, d \} \sim \mathcal{P}(q, H)$$

El sistema luego ejecuta **recuperación paralela multi-vista** a través de índices semánticos, léxicos y simbólicos, y fusiona los resultados mediante deduplicación basada en ID:

<table>
<tr>
<td width="50%">

**🔹 Consultas Simples**
- Búsqueda directa de hechos a través de una sola unidad de memoria
- Profundidad mínima de recuperación
- Tiempo de respuesta rápido

</td>
<td width="50%">

**🔸 Consultas Complejas**
- Agregación a través de múltiples eventos
- Profundidad de recuperación expandida
- Cobertura integral

</td>
</tr>
</table>

**📈 Resultado**: 43.24% de puntuación F1 con **30× menos tokens** que los métodos de contexto completo.

---

## 🚀 Aspectos Destacados del Rendimiento

### 📊 Resultados de Benchmark (LoCoMo)

<details>
<summary><b>🔬 Modelos de Alta Capacidad (GPT-4.1-mini)</b></summary>

| Tipo de Tarea | SimpleMem F1 | Mem0 F1 | Mejora |
|:----------|:------------:|:-------:|:-----------:|
| **MultiHop** | 43.46% | 30.14% | **+43.8%** |
| **Temporal** | 58.62% | 48.91% | **+19.9%** |
| **SingleHop** | 51.12% | 41.3% | **+23.8%** |

</details>

<details>
<summary><b>⚙️ Modelos Eficientes (Qwen2.5-1.5B)</b></summary>

| Métrica | SimpleMem | Mem0 | Notas |
|:-------|:---------:|:----:|:------|
| **F1 Promedio** | 25.23% | 23.77% | Competitivo con un modelo 99× más pequeño |

</details>

---

## 📦 Instalación

### 📋 Requisitos

- 🐍 Python 3.10
- 🔑 API compatible con OpenAI (OpenAI, Qwen, Azure OpenAI, etc.)

### 🛠️ Configuración

```bash
# 📥 Clonar repositorio
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem

# 📦 Instalar dependencias
pip install -r requirements.txt

# ⚙️ Configurar ajustes de API
cp config.py.example config.py
# Editar config.py con tu clave API y preferencias
```

### ⚙️ Ejemplo de Configuración

```python
# config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # o endpoint personalizado para Qwen/Azure

LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Recuperación de última generación
```

---

## ⚡ Inicio Rápido

### 🎓 Uso Básico

```python
from main import SimpleMemSystem

# 🚀 Inicializar sistema
system = SimpleMemSystem(clear_db=True)

# 💬 Agregar diálogos (Etapa 1: Compresión Estructurada Semántica)
system.add_dialogue("Alice", "Bob, let's meet at Starbucks tomorrow at 2pm", "2025-11-15T14:30:00")
system.add_dialogue("Bob", "Sure, I'll bring the market analysis report", "2025-11-15T14:31:00")

# ✅ Finalizar codificación atómica
system.finalize()

# 🔎 Consulta con recuperación consciente de la intención
answer = system.ask("When and where will Alice and Bob meet?")
print(answer)
# Salida: "16 November 2025 at 2:00 PM at Starbucks"
```

---

---

<div align="center">

# 🧠 Omni-SimpleMem: Multimodal Memory

**NEW** — SimpleMem now handles text, image, audio & video.

</div>

<table>
<tr>
<td align="center" width="140">📈 <b>+411%</b><br><sub>LoCoMo F1</sub></td>
<td align="center" width="140">📈 <b>+214%</b><br><sub>Mem-Gallery F1</sub></td>
<td align="center" width="140">⚡ <b>5.81 q/s</b><br><sub>3.5x faster</sub></td>
<td align="center" width="140">🧠 <b>4 modalities</b><br><sub>Text · Image · Audio · Video</sub></td>
</tr>
</table>

> 📖 Full documentation: [**Omni-SimpleMem →**](../../OmniSimpleMem/)

---

## 🔌 Servidor MCP *(text memory)*

SimpleMem está disponible como **servicio de memoria alojado en la nube** mediante el Model Context Protocol (MCP).

**🌐 Servicio en la Nube**: [mcp.simplemem.cloud](https://mcp.simplemem.cloud)

### Configuración Rápida

```json
{
  "mcpServers": {
    "simplemem": {
      "url": "https://mcp.simplemem.cloud/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

> 📖 Para instrucciones detalladas y guía de auto-alojamiento, consulta la [Documentación MCP](../../MCP/README.md)

---

---

## 🗺️ Hoja de Ruta

- [ ] Omni cross-session memory
- [ ] Omni MCP server
- [ ] Omni Docker support
- [ ] Omni PyPI package
- [ ] Streaming ingestion
- [ ] Multi-agent memory sharing

---

## 📊 Evaluación

```bash
# 🎯 Benchmark completo LoCoMo
python test_locomo10.py

# 📉 Evaluación de subconjunto (5 muestras)
python test_locomo10.py --num-samples 5
```

---

## 📝 Citación

```bibtex
@article{simplemem2025,
  title={SimpleMem: Efficient Lifelong Memory for LLM Agents},
  author={Liu, Jiaqi and Su, Yaofeng and Xia, Peng and Zhou, Yiyang and Han, Siwei and  Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2601.02553},
  year={2025},
  url={https://github.com/aiming-lab/SimpleMem}
}
```

---

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT** - consulta el archivo [LICENSE](../../LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- 🔍 **Modelo de Embeddings**: [Qwen3-Embedding](https://github.com/QwenLM/Qwen) - Rendimiento de recuperación de última generación
- 🗄️ **Base de Datos Vectorial**: [LanceDB](https://lancedb.com/) - Almacenamiento columnar de alto rendimiento
- 📊 **Benchmark**: [LoCoMo](https://github.com/snap-research/locomo) - Marco de evaluación de memoria de contexto largo
