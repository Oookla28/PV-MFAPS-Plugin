# 从反馈差异到动作计划

这份说明只讨论插件真正负责的那一段。它不替训练器决定 backbone，也不把 FIT 校准量包装成总体泛化定理。插件的作用是把一个训练过程中可以直接测量的差异，转换成一个有限、可检查、可执行的动作计划。

## 1. 两个读出头，回答一个具体问题

设 encoder 参数为 `ψ`，当前读出头为 `W_t`。在同一个 FIT batch 上，再用若干步内循环更新得到参考读出 `W̃_t`。两者都作用于同一个 loss：

$$
L(\psi,W)=\frac{1}{m}\sum_{i=1}^{m}
\ell\bigl(W z_\psi(x_i),y_i\bigr).
$$

当前头和参考头给 encoder 的梯度分别是

$$
g_t=\nabla_\psi L(\psi_t,W_t),
\qquad
\widetilde g_t=\nabla_\psi L(\psi_t,\widetilde W_t).
$$

插件记录两者的差

$$
e_t=g_t-\widetilde g_t.
$$

`e_t` 的含义很直接。假如它接近零，说明当前读出已经足够接近参考读出，继续精化读出对这一步 encoder 更新的影响有限。假如它很大，说明当前读出仍在改变 encoder 的学习方向。

为了比较不同 batch 和不同阶段，插件同时记录

$$
\kappa_t=
\frac{\lVert e_t\rVert_2}
{\lVert\widetilde g_t\rVert_2+\delta},
$$

其中 `δ` 是数值稳定项。`κ_t` 是训练轨迹上的诊断量。它需要和对应的 checkpoint、FIT batch、loss 定义一起保存。

## 2. 一个最小的二分类例子

考虑两个类别，候选动作只有 `(0,0)` 和 `(1,0.25)`。设

$$
A_1=A_2=1,
\qquad
\rho_1=0.9,
\qquad
\rho_2=0.7,
\qquad
\sigma_1=\sigma_2=0.1,
\qquad
n_1=n_2=10.
$$

对 native 动作 `(0,0)`，两类的代理风险都为 `1`。对精化动作 `(1,0.25)`，取 `ξ=1.01`，有

$$
U_1(1,0.25)
=0.75^2\times0.9^2
+0.25^2\times\frac{0.1^2}{10}\times1.01
\approx0.4557,
$$

$$
U_2(1,0.25)
=0.75^2\times0.7^2
+0.25^2\times\frac{0.1^2}{10}\times1.01
\approx0.2757.
$$

如果预算只允许精化一个类别，均匀权重下 DP 会选择类别 2。原因不是类别编号，也不是额外规则，而是类别 2 的收缩率更低，精化一步带来的偏差下降更大。这个例子展示了插件最基本的行为：**动作由风险表决定，风险表由校准量决定，预算决定最多能执行多少动作。**

## 3. 类别风险表

对类别 `c` 和动作 `a=(k,α)`，默认风险代理为

$$
U_c(k,\alpha)=(1-\alpha)^2 A_c\rho_c^{2k}+\alpha^2\frac{\sigma_c^2}{n_c}\xi^k.
$$

第一项随 `k` 变化，描述当前读出误差在内循环中的衰减。第二项包含 `σ_c`、样本支持 `n_c` 和增长因子 `ξ`，用来表示参考反馈的波动和精化动作的稳定性代价。

这里的 `A_c`、`ρ_c`、`σ_c` 和 `n_c` 都是 FIT 校准量。插件把它们当作一张风险表的输入。若 host trainer 已经测得更合适的逐动作风险，可以直接绕过默认公式，将测得的风险值交给 DP。

## 4. 带预算的离散优化

令 `q_c` 表示类别权重，`ν_c` 表示尾部强调项，动作成本为 `d_c(a_c)`，总预算为 `B`。插件求解

$$
\min_{a_1,\ldots,a_C}
\sum_{c=1}^{C}
\left(\frac{\mu}{C}+\nu_c\right)U_c(a_c)
\quad\text{s.t.}\quad
\sum_{c=1}^{C}d_c(a_c)\le B.
$$

动作不是连续变量，而是来自有限集合

$$
\mathcal A=\{(0,0),(1,\alpha_1),(2,\alpha_2),(4,\alpha_4),\ldots\}.
$$

这种离散化有两个好处。第一，训练器只会收到明确的合法动作。第二，求解器可以把最终动作、成本和目标值写进证书，读者能够重新计算，而不需要猜一个连续优化器在内部做了什么。

## 5. Pareto frontier 动态规划

处理完前 `j` 个类别后，一个状态写成

$$
S_j=(b_j,r_j,s_j,\pi_j),
$$

其中 `b_j` 是量化成本，`r_j` 是累计风险，`s_j` 是加权目标，`π_j` 是已经选择的动作前缀。

给第 `j+1` 个类别加入动作 `a` 后，状态变为

$$
S_{j+1}=
\bigl(b_j+d_{j+1}(a),
      r_j+U_{j+1}(a),
      s_j+w_{j+1}U_{j+1}(a),
      \pi_j\cup\{a\}\bigr).
$$

如果新成本超过 `B`，状态直接丢弃。对于相同的量化成本，如果状态 `S` 在风险和目标两个坐标上都不优于状态 `S'`，则 `S` 被支配，也可以丢弃。剩下的状态就是 Pareto frontier。

代码中的成本先乘以 `budget_scale`，再向上取整。向上取整会让求解器稍微保守，但可以保证返回的量化方案不会因为浮点误差超过原始预算。最终证书记录：

- `k` 和 `α` 动作向量
- 量化预算与实际成本
- 目标值和风险总量
- 每个类别的局部风险
- 每一阶段保留下来的 frontier 数量

因此，“离散最优”只针对给定的风险表、动作集合和量化成本问题。它不自动说明校准量在总体数据上的统计性质。

## 6. Fused 执行为什么可以省计算

Legacy 路径把 native、reference 和额外 solver action 分开执行，可能重复计算相同 batch 的 encoder 特征。Fused 路径先计算一次

$$
H=\mathrm{Encoder}(X),
$$

再把 `H` 交给多个读出分支

$$
Z_{\mathrm{native}}=W_tH,
\qquad
Z_{\mathrm{ref}}=\widetilde W_tH.
$$

这两个式子说明了比较口径。Fused 可以减少物理 encoder forward，但不能悄悄改变动作计划。完整对照应同时记录

$$
\text{logical actions}_{\mathrm{Legacy}}
=\text{logical actions}_{\mathrm{Fused}},
$$

以及

$$
\text{physical forwards}_{\mathrm{Fused}}
<\text{physical forwards}_{\mathrm{Legacy}}.
$$

模块里的 `ExecutionTrace` 把这两类计数分开保存。host trainer 还需要在 Legacy 和 Fused 之间重置模型、优化器和随机种子，并使用同步后的 CUDA 计时。

## 7. 一次完整接入应留下什么

一个可复查的 host 运行至少应保存：

1. FIT 校准数据和来源哈希。
2. 类别顺序、动作网格和预算。
3. 选择出的 `(k,α)` 计划和 DP 证书。
4. native、reference、solver 的逻辑计数。
5. encoder 的物理 forward/backward 计数。
6. FIT 指标、evaluation 指标、FLOPs 和 wall-clock。
7. 冻结 checkpoint 的来源和配置哈希。

这样，插件只负责它真正拥有的部分，训练器仍然可以按自己的方式管理数据、模型和评估流程。
