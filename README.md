# dsh-compaction-opencode

把 **OpenCode 的上下文压缩策略**移植成 DeepSeek Harness 的压缩引擎（`ctx.compaction` 的一个实现）。

移植源：`opencode/packages/core/src/session/compaction.ts`（963 行），辅以 `session/token.ts`
与 `runner/to-llm-message.ts`。宿主契约取自 `@deepseek-ai/dsh-compaction`，事务机制对齐
`@deepseek-ai/dsh-compaction-basic`（DSH 自带的参考实现）。

引擎是**纯宿主插件**，没有 client 半侧：压缩不往 UI 注入任何东西。

---

## 1. 移植了什么

### 1.1 预算策略（OpenCode 的判据，逐字对照）

| 名称 | 值 | 作用 |
| --- | --- | --- |
| `calculateCeiling(window, buffer)` | — | 压缩天花板：显式 `buffer` 则 `window - buffer`；否则 `window - max(floor(window*0.1), window>=32000 ? 16000 : 0)` |
| `RESERVE_MIN` | `16_000` | 大窗口（≥ 2×）才启用的最低预留 |
| `UNKNOWN_WINDOW` | `200_000` | 窗口未知时的兜底 |
| `SHRINK_STEPS` | `[0.7, 0.5, 0.35]` | 被拒后的逐级收缩阶梯 |
| `keep` | `15_000` | 逐字保留的近期尾巴（token） |

触发判据 `due`：最后一条消息已是完成的检查点则跳过；最近一次**实测** prompt 早于最近一次
检查点则跳过；否则 `estimateContext >= ceiling`。

实测口径沿用 OpenCode：只有「同一 `providerID`、无 error、且
`input + cache.read + cache.write > 0`」的 assistant 消息才算锚点，其
`input + cache.read + cache.write + output + reasoning` 作为实测值，其后的消息按 `/4` 估算。

### 1.2 保留边界（`splitConversation` / `recentStart`）

从尾部向前累加到 `keep`，然后**回退到一条 user 消息**作为保留段起点（`oldestToDrop` +
`userBoundary`）。全都能装下时只保留最后一轮对话；再无更老的轮次则判定「无可压缩」。

### 1.3 摘要契约（结构化模板 + 单次纠偏）

`SUMMARY_TEMPLATE` 要求 `## Objective` / `## Requirements` / `## Decisions` /
`## Work State`（含 `### Completed` / `### Active` / `### Blocked`）/ `## Next Move` /
`## Relevant Files`（≤15 条）/ `## Important Context`。

`SUMMARY_RULES` 要求：单行短句；保留**逐字**的路径、符号、命令、报错串、URL、标识符；
只结转**未答复**的用户请求且保留原措辞；保留工作流状态（uncommitted / committed / pushed /
review / merged）；**不得提及压缩本身**。

`filled(text)` 校验是否至少出现一个模板标题；不通过则追加一次 `NUDGE` 重试（**只重试一次**），
仍不通过就 fail-closed：有内容但格式不符 → `Compaction summary did not match the required template`，
无内容 → `Compaction produced no summary`。

`buildPrompt(update, legacy)` 区分三种提示：首次总结、更新既有检查点、以及旧版检查点（含
`## Additional Context` 时要求改写成新格式）。

### 1.4 检查点线格（`runner/to-llm-message.ts`）

替换消息渲染为单条 user 消息：

```
<conversation-checkpoint>
<summary>…结构化摘要…</summary>
<recent-context>…逐字保留的近期对话…</recent-context>
</conversation-checkpoint>
```

### 1.5 压缩失败时的兜底（`flattenAndDropOldest`）

装不下时把保留段**展平为文本**、按「一轮对话」为单位从最老处整轮丢弃，并加一行
`[N older exchanges omitted]`。宁可丢整轮，也不切断工具调用与其结果。

### 1.6 工具输出与媒体计价

`truncateToolOutput` 按**码点**截断到 `TOOL_OUTPUT_MAX_CHARS = 1_250` 并追加 `[truncated]`
（按码点是为了不劈开代理对）。`image/*` 计 `1_500`，`application/pdf` 计 `2_000`。

---

## 2. 没移植什么（及原因）

| 未移植 | 原因 |
| --- | --- |
| drizzle / SQLite 投影层 | DSH 用 append-only 的 `Session` 日志 + `surface`，没有对应物 |
| Effect 运行时 | DSH 用 Cordis 的 `Service` / `ctx.on` / `ctx.waterfall` |
| OpenCode 事件总线 | 换成 DSH 的 `agent/pre-step`、`agent/request-error`、`compaction/summary-error` |
| provider 原生压缩窗口 | 与 DSH 的 `surface` 模型不兼容，且需要 provider 私有协议 |
| 模型目录 / 计费表 | DSH 的模型信息来自 `ctx.llm.resolveModelInfo` |
| `payload-too-large` 单独分支 | DSH 只有 `CONTEXT_WINDOW_EXCEEDED` 一个码，且尺寸判定是正则的 |

---

## 3. DSH 适配点（与 OpenCode 的实质差异）

1. **事务协议**。OpenCode 没有日志级锁，DSH 契约要求成功路径是：
   `compaction/start`（取得锁）→ 摘要请求 → `compaction/summary`（记录影价）→
   用一条 `user/message` 以 `surfaceOp: replace` **替换**选定区间（唯一的 surface 变更）→
   `compaction/end`（释放锁）。失败时在 catch 里做**恰好一次** `compaction/end` 带 `error`。
   这段机制来自 `dsh-compaction-basic`，不是 OpenCode。

2. **阈值更严**。OpenCode 把回复额度排除在窗口之外；DSH 把路由的 `maxTokens` 算进同一窗口。
   因此压力预算取两者更小值：

   ```
   threshold = min( calculateCeiling(contextWindow, buffer),
                    contextWindow - reservedCompletionTokens - headroomTokens )
   ```

   不加这道约束就会按「不会溢出」的乐观值触发压缩。

3. **`recentMode`**。OpenCode 只能把保留段写进 `<recent-context>`（它没有 surface 概念）。
   DSH 的 surface 允许保留段继续作为**真实节点**存在，所以默认是 `surface`（不写
   `<recent-context>`），`text` 才复刻 OpenCode 的原行为。

4. **恢复钩子**。OpenCode 用自己的 hook；DSH 用 `compaction/summary-error` waterfall，
   监听方返回 `true` 表示已记录真实进展，引擎才重新准备并重试。

5. **工具配对边界**。抽取边界必须不被「未答复的工具调用」跨越，用
   `toolPairingBalancedBefore/After` 校验；这是 DSH 契约的硬要求。

---

## 4. 配置

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `auto` | `true` | 注册自动压缩（`agent/pre-step` 压力 + `agent/request-error` 溢出恢复） |
| `buffer` | 未设 | 显式预留；不设则走 OpenCode 的 `10% / 16000` 规则 |
| `keep` | `15000` | 逐字保留的近期尾巴（token） |
| `headroomTokens` | `0` | 额外压低阈值的余量 |
| `maxTokens` | `32000` | 摘要请求的输出上限 |
| `recentMode` | `surface` | `surface` \| `text` |
| `summarizationProvider` / `summarizationModel` | 空 | **必须成对**；省略则用最近一次真实请求的 provider/model |
| `compactionRetries` | `1` | 单次调用内允许的重复压缩轮数 |
| `maxOverflowRetries` | `1` | 溢出后的恢复重试次数 |

配置非法（未知键、非整数、半对、`keep >= threshold`）一律抛错并 fail-closed；
压力阈值的配置错误按 provider/model 只告警一次。

### 实测记录（live 会话，2026-10-13）

在真实 DSH 会话里把阈值临时压低（改 profile patch 的 `buffer`/`keep`）跑出来的轨迹。
logger 名是 `opencode-compaction-engine`（Cordis 由 entry id 派生），与
`compaction-basic` 的 `basic-compaction-engine` 可区分——**故可判定生效后端**。

| 观察 | 数值 |
| --- | --- |
| 触发点 | `agent/pre-step`，在真实会话里被触发 |
| surface 走向 | 203202 → 159513 → 114089 → 107340 tokens |
| 每次尝试轮数 | 2（`compactionRetries: 1` 的语义） |
| 由观测反推 | `contextWindow = 1_000_000`；`buffer` 路径确实压过 messageBudget |

两条**预期行为**（都不是缺陷，均由人为压低阈值暴露）：

1. `The compaction request cannot be reduced further without losing the latest
   exchange or checkpoint` —— 摘要请求的预算小于**单个最新 exchange**时，
   `flattenAndDropOldest` 按设计拒绝丢掉最新 exchange，于是 fail-closed 并继续本轮。
2. `still above the pressure threshold after N attempt(s)` —— 保留段自身就超过阈值
   （`keep` 与 threshold 太近）时无法收敛。

**一个重要的设计属性**：保留段按 OpenCode 的 `recentStart` 语义**必须从一条
user message 开始**（`lib/index.js` 的 `_select`）。因此**单个超长 turn 无法被摘要掉**，
压缩后的 surface 会稳定在该 turn 的体量上——这是移植保真的行为，不是缺陷。
配 `keep`/`buffer` 时要留出比「当前 turn」更大的余量。

---

## 5. 安装

本插件依赖 `@deepseek-ai/*` 的宿主包，这些包由 DSH Desktop 随包携带，通过
`resources/host-module-fallback.mjs` 的 resolve 钩子解析——**插件目录自身不需要
`node_modules`**。这与 `dsh-humanizer-zh` 的安装形态一致。

### 路线 A：手工安装（对照 `dsh-humanizer-zh`，推荐用于本地开发）

1. 把本目录放到 profile 的 `node_modules` 下：

   ```
   <harness>\profiles\web\node_modules\dsh-compaction-opencode\
   ```

2. 在 `<harness>\profiles\web\cordis.patch.yml` 末尾追加：

   ```yaml
   - insert:
       - id: dsh-compaction-opencode
         name: 'dsh-compaction-opencode'
   ```

3. 确认没有第二个 `ctx.compaction` 实现。`include:compaction-basic` 当前是
   `enabled: false`，保持如此即可；两个实现同时启用会争抢同一服务。

### 路线 B：纳入 profile bundles（对照商场上架插件）

放进 `profiles\.generations\live\<pkg>+<ver>+<hash>\node_modules\<pkg>`，软链到
`profiles\web\node_modules`，并在 `profiles\web\package.json` 的 `dependencies`、
`pnpm.overrides`、`dsh.profile.bundles` 三处登记。

### 路线 C：重装

先确认包是完整的。`package.json#files` 现在发 **7 个文件**，`dsh.plugin.json` 与
`CHANGELOG.md` 曾经不在列表里（前者因此被静默丢弃）：

```sh
npm pack --ignore-scripts     # -> dsh-compaction-opencode-0.2.2.tgz，7 files / 28.5 kB
```

重装**必须先清掉旧版**，否则会同时存在两个 `ctx.compaction` 实现：

1. 删掉旧 generation 目录
   `<harness>\profiles\.generations\live\dsh-compaction-opencode+0.2.0-rc.2+<旧hash>\`
2. 从 `profiles\web\package.json` 移除三处旧登记：
   `dependencies["dsh-compaction-opencode"]`、`dsh.profile.bundles` 里的条目、
   `pnpm.overrides["dsh-compaction-opencode"]`
3. 解包后按路线 B 装入新 generation，并把三处登记写回 `0.2.2` 与新的 generation id
4. 重启 DSH Desktop（新增 `insert` 行需要重启才生效）

若走图形界面的插件管理安装，第 1–3 步由它代办。

### 注意

- 新增 `insert` 行通常需要**重启 DSH Desktop** 才生效；重启会中断当前会话。
- `billion-context`（ACP 代理层压缩）与本引擎**不冲突**：一个在网关/代理层压缩请求，
  一个在宿主内压缩会话历史。但两者叠加会各自独立触发，需要留意整体压缩强度。

---

## 6. 验证状态

| 套件 | 规模 | 覆盖 |
| --- | --- | --- |
| `.scratch/smoke/run.mjs` | **150 项通过** | 用 stub 宿主跑完整事务 S1–S11：纯函数、压力触发、区间选择、事务锁、失败路径、`compactNow`、`recentMode`、模板纠偏、溢出恢复、阴影价校验 |
| `.scratch/verify-host.mjs` | **93 项通过** | 通过 `host-module-fallback` 加载**真实** `@deepseek-ai/*` 包：继承真实 `CompactionEngine`、`Config` 校验、真实 `createUserMessage` / `ContentBlock` / `compactCheckpointSource` 走通渲染与检查点往返；**真实端到端压缩**——真 `Context` 上的真引擎、真 `Session` 作 surface、真 `BlockAssembler` 消费真 `StreamChunk` 协议，断言锁/影价/surface 代际/线格/压缩后回到阈值以下；**真触发器路径**——通过真 `waterfall` 分发 `agent/pre-step`（压力）与 `agent/request-error`（`CONTEXT_WINDOW_EXCEEDED` 溢出），并反向断言低于阈值不压、其它错误码不接管；以及 **live 实测踩出来的三条守卫** 与 `compactNow` 的真包端到端 |
| 打包产物（`npm pack` 解包后） | **243 项通过** | 两个套件都支持 `OC_PLUGIN_URL`，指向解包目录即可运行。这条证明 `files` 发全了运行时所需的一切，而不是只证明源码树能跑 |

两个套件都在仓库根的 `.scratch/` 下，可直接 `node` 运行。

### 0.2.1：修掉一个测试侧的盲点

`.scratch/smoke/stubs/dsh-llm.mjs` 里的 `BlockAssembler` 此前接受一套**伪造的块协议**
（`{type:"text"}` 与 `{type:"finish",finish}`），并且**静默丢弃**所有它不认识的 chunk。
真协议是 `block-start` / `text-delta` / `reasoning-delta` / `tool-call-delta` /
`block-end` / `usage` / `finish`(`reason`)，且真 `BlockAssembler.push` 的 `default`
分支是 `assertNever`——未知块必须抛错。

后果是这套测试**不可能发现流协议写错**。现已按真协议重写 stub（未知块同样抛错），
`makeHost` 的脚本化 provider 改为产出真 chunk，并把真实端到端压缩固化进
`verify-host.mjs`。插件本身无需改动：它把 chunk 直接交给真 `BlockAssembler`，
所以 150 项 smoke 在真协议下依然全绿。

### 0.2.2：补上 live 实测踩出来的守卫

2026-10-13 在真实会话里压低阈值强制触发压缩，暴露出三条**已实际执行却零测试覆盖**的
代码路径。现在都固化为真包回归（`verify-host.mjs`，75 → 93）：

| 守卫 | 触发条件 | 断言 |
| --- | --- | --- |
| `COMPACTION_UNSHRINKABLE` | 摘要预算装不下单个最新 exchange | 警告 + **surface 保持不动** |
| `COMPACTION_NOT_SMALLER` | 模型返回的摘要不比被替换内容更短 | 警告 + **surface 保持不动** |
| `still above the pressure threshold` | 保留段自身就超过阈值 | 走满 `compactionRetries + 1` 次后停止并警告 |

同时补上 `compactNow`（`/compact` 的唯一路径）的真包端到端，并固化它的两条契约：
**会话有开启中的 turn 时拒绝**（`ManualCompactionError`，`code: "busy"`，`lib/index.js` L738），
以及**空闲会话即使低于压力阈值也能手动压**（这是手动压缩的意义所在）。

「保留段必须从 user message 开始」也写成断言了（`_select`，L1051–1054）——它是
**单个超长 turn 设下 floor** 的直接原因，将来不应被当成 bug「修」掉。

两个写夹具时踩到的陷阱，记录下来以免下次重犯：

1. `COMPACTION_NOT_SMALLER` 的摘要必须**决定性地**大于被替换内容。只是「差不多大」时，
   第一次尝试会成功提交、守卫只落在重试上——那样测的是错的东西（surface 代际会前进）。
2. `still above` 需要一个**每次重试都更短**的桩摘要。因为 `_select` 在重试时会重新选中
   刚写入的检查点，定长摘要会先撞上 `NOT_SMALLER` 抛出，循环根本走不到收敛警告。

---

## 7. 许可

MIT。移植的算法与提示词源自 OpenCode（MIT）。
