<div align="center">

<img alt="simplemem_logo" src="https://github.com/user-attachments/assets/6ea54ad1-e007-442c-99d7-1174b10d1fec" width="450">

<div align="center">

## Mémoire à Vie Efficace pour les Agents LLM — Text & Multimodal

<small>Stockez, compressez et récupérez des mémoires à long terme grâce à la compression sémantique sans perte. Compatible avec Claude, Cursor, LM Studio et bien d'autres.</small>

</div>

<p><b>Fonctionne avec toute plateforme d'IA supportant MCP ou l'intégration Python</b></p>

<table>
<tr>
<td align="center" width="100">
  <a href="https://www.anthropic.com/claude">
    <img src="https://cdn.simpleicons.org/claude/D97757" width="48" height="48" alt="Claude Desktop" />
  </a><br/>
  <sub><a href="https://www.anthropic.com/claude"><b>Claude Desktop</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://cursor.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/cursor/FFFFFF">
      <img src="https://cdn.simpleicons.org/cursor/000000" width="48" height="48" alt="Cursor" />
    </picture>
  </a><br/>
  <sub><a href="https://cursor.com"><b>Cursor</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://lmstudio.ai">
    <img src="https://github.com/lmstudio-ai.png?size=200" width="48" height="48" alt="LM Studio" />
  </a><br/>
  <sub><a href="https://lmstudio.ai"><b>LM Studio</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://cherry-ai.com">
    <img src="https://github.com/CherryHQ.png?size=200" width="48" height="48" alt="Cherry Studio" />
  </a><br/>
  <sub><a href="https://cherry-ai.com"><b>Cherry Studio</b></a></sub>
</td>
<td align="center" width="100">
  <a href="https://pypi.org/project/simplemem/">
    <img src="https://cdn.simpleicons.org/pypi/3775A9" width="48" height="48" alt="PyPI" />
  </a><br/>
  <sub><a href="https://pypi.org/project/simplemem/"><b>Paquet PyPI</b></a></sub>
</td>
<td align="center" width="100">
  <sub><b>+ Tout Client<br/>MCP</b></sub>
</td>
</tr>
</table>

<div align="center">

<br/>

[🇨🇳 中文](./README.zh-CN.md) •
[🇯🇵 日本語](./README.ja.md) •
[🇰🇷 한국어](./README.ko.md) •
[🇪🇸 Español](./README.es.md) •
[🇫🇷 **Français**](./README.fr.md) •
[🇩🇪 Deutsch](./README.de.md) •
[🇧🇷 Português](./README.pt-br.md)<br/>
[🇷🇺 Русский](./README.ru.md) •
[🇸🇦 العربية](./README.ar.md) •
[🇮🇹 Italiano](./README.it.md) •
[🇻🇳 Tiếng Việt](./README.vi.md) •
[🇹🇷 Türkçe](./README.tr.md)

<br/>

[![Project Page](https://img.shields.io/badge/🎬_DÉMO_INTERACTIVE-Visiter_le_Site-FF6B6B?style=for-the-badge&labelColor=FF6B6B&color=4ECDC4&logoColor=white)](https://aiming-lab.github.io/SimpleMem-Page)

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
  <a href="https://discord.gg/KA2zC32M"><img src="https://img.shields.io/badge/Discord-Rejoindre-5865F2?style=flat&labelColor=555&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="../../fig/wechat_logo3.JPG"><img src="https://img.shields.io/badge/WeChat-Groupe-07C160?style=flat&labelColor=555&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

<br/>

[Aperçu](#-aperçu) • [Démarrage Rapide](#-démarrage-rapide) • [Serveur MCP](#-serveur-mcp) • [Évaluation](#-évaluation) • [Citation](#-citation)

</div>

</div>

<br/>

## 🔥 Actualités

- **[02/09/2026]** 🚀 **Mémoire Cross-Session disponible - 64% plus performant que Claude-Mem !** SimpleMem prend désormais en charge la **mémoire persistante entre les conversations**. Sur le benchmark LoCoMo, SimpleMem atteint une **amélioration de 64%** par rapport à Claude-Mem. Vos agents peuvent maintenant se souvenir automatiquement du contexte, des décisions et des apprentissages des sessions précédentes. [Voir Documentation Cross-Session →](../../cross/README.md)
- **[01/20/2026]** **SimpleMem est maintenant disponible sur PyPI !** 📦 Installez directement via `pip install simplemem`. [Voir le Guide d'Utilisation →](../PACKAGE_USAGE.md)
- **[01/19/2026]** **Stockage de mémoire locale ajouté à SimpleMem Skill !** 💾
- **[01/18/2026]** **SimpleMem supporte maintenant Claude Skills !** 🚀 Utilisez SimpleMem dans claude.ai pour une mémoire à long terme entre les conversations.
- **[01/14/2026]** **Le serveur MCP SimpleMem est EN LIGNE et Open Source !** 🎉 Service de mémoire cloud sur [mcp.simplemem.cloud](https://mcp.simplemem.cloud). [Voir la Documentation MCP →](../../MCP/README.md)
- **[01/08/2026]** 🔥 Rejoignez notre [Discord](https://discord.gg/KA2zC32M) et [groupe WeChat](../../fig/wechat_logo3.JPG) !
- **[01/05/2026]** L'article SimpleMem a été publié sur [arXiv](https://arxiv.org/abs/2601.02553) !

---

## 🌟 Aperçu

<div align="center">
<img src="../../fig/Fig_tradeoff.png" alt="Compromis Performance vs Efficacité" width="900"/>

*SimpleMem atteint un score F1 supérieur (43.24%) avec un coût minimal en tokens (~550).*
</div>

**SimpleMem** est un cadre de mémoire efficace basé sur la **compression sémantique sans perte** qui répond au défi fondamental de la **mémoire à long terme efficace pour les agents LLM**. Contrairement aux systèmes existants, SimpleMem maximise la **densité d'information** et l'**utilisation des tokens** à travers un pipeline en trois étapes :

<table>
<tr>
<td width="33%" align="center">

### 🔍 Étape 1
**Compression Structurée Sémantique**

Distille les interactions non structurées en unités de mémoire compactes avec indexation multi-vue

</td>
<td width="33%" align="center">

### 🗂️ Étape 2
**Synthèse Sémantique en Ligne**

Intègre instantanément le contexte connexe en représentations abstraites unifiées pour éliminer la redondance

</td>
<td width="33%" align="center">

### 🎯 Étape 3
**Planification de Récupération Consciente de l'Intention**

Infère l'intention de recherche pour déterminer dynamiquement la portée de récupération

</td>
</tr>
</table>

<div align="center">
<img src="../../fig/Fig_framework.png" alt="Architecture SimpleMem" width="900"/>
</div>

---

### 🏆 Comparaison des Performances

<div align="center">

| Modèle | ⏱️ Construction | 🔎 Récupération | ⚡ Total | 🎯 F1 Moyen |
|:------|:--------------------:|:-----------------:|:-------------:|:-------------:|
| A-Mem | 5140.5s | 796.7s | 5937.2s | 32.58% |
| LightMem | 97.8s | 577.1s | 675.9s | 24.63% |
| Mem0 | 1350.9s | 583.4s | 1934.3s | 34.20% |
| **SimpleMem** ⭐ | **92.6s** | **388.3s** | **480.9s** | **43.24%** |

</div>

---

## 📦 Installation

```bash
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem
pip install -r requirements.txt
cp config.py.example config.py
```

---

## ⚡ Démarrage Rapide

```python
from main import SimpleMemSystem

system = SimpleMemSystem(clear_db=True)
system.add_dialogue("Alice", "Bob, let's meet at Starbucks tomorrow at 2pm", "2025-11-15T14:30:00")
system.add_dialogue("Bob", "Sure, I'll bring the market analysis report", "2025-11-15T14:31:00")
system.finalize()

answer = system.ask("When and where will Alice and Bob meet?")
print(answer)
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

## 🔌 Serveur MCP *(text memory)*

**🌐 Service Cloud** : [mcp.simplemem.cloud](https://mcp.simplemem.cloud)

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

> 📖 [Documentation MCP](../../MCP/README.md)

---

---

## 🗺️ Feuille de Route

- [ ] Omni cross-session memory
- [ ] Omni MCP server
- [ ] Omni Docker support
- [ ] Omni PyPI package
- [ ] Streaming ingestion
- [ ] Multi-agent memory sharing

---

## 📊 Évaluation

```bash
python test_locomo10.py
python test_locomo10.py --num-samples 5
```

---

## 📝 Citation

```bibtex
@article{simplemem2025,
  title={SimpleMem: Efficient Lifelong Memory for LLM Agents},
  author={Liu, Jiaqi and Su, Yaofeng and Xia, Peng and Zhou, Yiyang and Han, Siwei and  Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2601.02553},
  year={2025},
  url={https://github.com/aiming-lab/SimpleMem}
}
```

## 📄 Licence

Ce projet est sous **Licence MIT** - voir le fichier [LICENSE](../../LICENSE).

## 🙏 Remerciements

- 🔍 **Modèle d'Embeddings** : [Qwen3-Embedding](https://github.com/QwenLM/Qwen)
- 🗄️ **Base de Données Vectorielle** : [LanceDB](https://lancedb.com/)
- 📊 **Benchmark** : [LoCoMo](https://github.com/snap-research/locomo)
