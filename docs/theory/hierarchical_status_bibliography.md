# neurophase — канонічна ієрархічно-статусна бібліографія (Research-Grade)

**Версія:** 2026-04-11  
**Стандарт:** laboratory-grade documentation (реплікованість, фальсифікованість, трасованість до коду)  
**Мета:** забезпечити рівень аргументації, сумісний з R&D-практиками рівня DeepMind/OpenAI.

---

## 1) Політика відбору джерел (evidence governance)

### 1.1 Критерії включення

Джерело входить у канон лише якщо виконує **всі** умови:
1. **Первинність:** peer-reviewed article / академічна монографія / офіційний стандарт.
2. **Інституційна вага:** Nature-family, Neuron/Trends, PNAS, провідні видавництва (OUP/MIT/CUP/Springer).
3. **Методична придатність:** містить операціоналізовані змінні, які можна вбудувати в `neurophase`.
4. **Відтворюваність:** є чіткий дизайн вимірювання або математичне формулювання.
5. **Фальсифікованість:** дозволяє сформулювати критерій спростування для продуктового claim.

### 1.2 Критерії виключення

Виключаються:
- нерецензовані opinion/blog-джерела як первинна база;
- «перекази переказів» без привʼязки до оригінального DOI/книги;
- модні наративи без стійкої емпіричної опори;
- джерела, де неможливо визначити статистичні припущення або межі валідності.

---

## 2) Ієрархія статусу джерел

- **S-tier (Foundational Canon):** визначає онтологію системи та базові закони динаміки.
- **A-tier (Mechanistic & High-Evidence):** повʼязує теорію з нейрокогнітивними механізмами керування.
- **B-tier (Method & Validation):** дає обчислювані метрики, протоколи перевірки, статистичну дисципліну.

---

## 3) Канонічний список джерел

## S-tier — фундамент емерджентності, синхронізації, предиктивної обробки

1. **Haken, H. (1983). _Synergetics: An Introduction_. Springer.**
   - Роль: формальна рамка самоорганізації (order parameters, control parameters).
   - Внесок у neurophase: теоретичний базис для «оркестрації умов», а не жорсткого програмування.

2. **Kelso, J. A. S. (1995). _Dynamic Patterns: The Self-Organization of Brain and Behavior_. MIT Press.**
   - Роль: координаційна динаміка мозок–поведінка.
   - Внесок у neurophase: пояснення фазових переходів когнітивних режимів.

3. **Strogatz, S. H. (2003). _Sync: The Emerging Science of Spontaneous Order_. Hyperion.**
   - Роль: канон синхронізації у складних системах.
   - Внесок у neurophase: концептуальна опора для фазової узгодженості та мережевої когерентності.

4. **Friston, K. (2010). The free-energy principle: a unified brain theory? _Nature Reviews Neuroscience_, 11(2), 127–138.**
   - DOI: `10.1038/nrn2787`
   - Роль: unified framework predictive processing.
   - Внесок у neurophase: `R(t)` як інженерна апроксимація prediction-error pressure.

5. **Clark, A. (2016). _Surfing Uncertainty: Prediction, Action, and the Embodied Mind_. Oxford University Press.**
   - Роль: bridge між теорією predictive mind і практикою дії/інтерфейсу.
   - Внесок у neurophase: дизайн active verification loops перед high-impact action.

6. **Goldberger, A. L. et al. (2002). Fractal dynamics in physiology: Alterations with disease and aging. _PNAS_, 99(Suppl 1), 2466–2472.**
   - DOI: `10.1073/pnas.012579499`
   - Роль: фундамент complexity physiology.
   - Внесок у neurophase: обґрунтування нелінійних/багатомасштабних індикаторів стану.

## A-tier — механізми виконавчого контролю, стресу, ритмів та автономної регуляції

7. **Miyake, A. et al. (2000). The unity and diversity of executive functions... _Cognitive Psychology_, 41(1), 49–100.**
   - DOI: `10.1006/cogp.1999.0734`
   - Роль: декомпозиція executive function на окремі компоненти.
   - Внесок у neurophase: окреме керування інгібіцією/оновленням/перемиканням.

8. **Arnsten, A. F. T. (2009). Stress signalling pathways that impair prefrontal cortex structure and function. _Nature Reviews Neuroscience_, 10(6), 410–422.**
   - DOI: `10.1038/nrn2648`
   - Роль: нейробіологія деградації контролю під стресом.
   - Внесок у neurophase: thresholds для `execution_gate` і `pacing`.

9. **Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013). The expected value of control. _Neuron_, 79(2), 217–240.**
   - DOI: `10.1016/j.neuron.2013.07.007`
   - Роль: обчислювальна теорія allocation of control.
   - Внесок у neurophase: політика «ціна когнітивного зусилля vs очікувана користь».

10. **Engel, A. K., Fries, P., & Singer, W. (2001). Dynamic predictions... _Nature Reviews Neuroscience_, 2(10), 704–716.**
    - DOI: `10.1038/35094565`
    - Роль: функціональна значущість нейронної синхронізації.
    - Внесок у neurophase: біологічна валідність oscillatory coupling.

11. **Cavanagh, J. F., & Frank, M. J. (2014). Frontal theta as a mechanism for cognitive control. _Trends in Cognitive Sciences_, 18(8), 414–421.**
    - DOI: `10.1016/j.tics.2014.04.012`
    - Роль: ритмічний механізм контролю й моніторингу конфлікту.
    - Внесок у neurophase: real-time control-state inference.

12. **Buzsáki, G. (2006). _Rhythms of the Brain_. Oxford University Press.**
    - Роль: канон нейроритмів і часової організації обчислень мозку.
    - Внесок у neurophase: multi-band оркестрація стану.

13. **Thayer, J. F., & Lane, R. D. (2000). A model of neurovisceral integration... _Journal of Affective Disorders_, 61(3), 201–216.**
    - DOI: `10.1016/S0165-0327(00)00338-4`
    - Роль: звʼязок автономної регуляції і top-down control.
    - Внесок у neurophase: HRV як канал оцінки адаптивності контролю.

14. **Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. _Frontiers in Public Health_, 5, 258.**
    - DOI: `10.3389/fpubh.2017.00258`
    - Роль: практичний словник метрик HRV.
    - Внесок у neurophase: стандартизована інтерпретація RMSSD/HF/SDNN.

## B-tier — метрики синхронії, мережеві підходи, статистична доброчесність

15. **Lachaux, J.-P. et al. (1999). Measuring phase synchrony in brain signals. _Human Brain Mapping_, 8(4), 194–208.**
   - DOI: `10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C`
   - Роль: методологічна основа phase synchrony.
   - Внесок у neurophase: базова формалізація PLV-обчислень.

16. **Bassett, D. S., & Sporns, O. (2017). Network neuroscience. _Nature Neuroscience_, 20(3), 353–364.**
   - DOI: `10.1038/nn.4502`
   - Роль: мережеве мислення про мозок як динамічний граф.
   - Внесок у neurophase: локальні сигнали → глобальний стан мережі.

17. **Deco, G., Jirsa, V., & McIntosh, A. R. (2011). Emerging concepts for the dynamical organization of resting-state activity. _Nature Reviews Neuroscience_, 12(1), 43–56.**
   - DOI: `10.1038/nrn2961`
   - Роль: макрорівнева динаміка станів мозку.
   - Внесок у neurophase: state-space інтерпретація режимів ризику/стабільності.

18. **Cohen, J. (1988). _Statistical Power Analysis for the Behavioral Sciences_ (2nd ed.). Routledge.**
   - Роль: planning power та ефект-розміри.
   - Внесок у neurophase: дисципліна планування експериментів до запуску.

19. **Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate... _Journal of the Royal Statistical Society. Series B_, 57(1), 289–300.**
   - Роль: контроль множинного тестування.
   - Внесок у neurophase: зниження ризику хибнопозитивних висновків.

---

## 4) Рекурсивна матриця трасованості (Claim → Mechanism → Metric → Falsification)

| Продуктовий claim | Механізм | Метрика | Критерій спростування |
|---|---|---|---|
| Оркестрація стану підвищує якість рішень | Самоорганізація та синхронізація | PLV/мережева когерентність + точність рішень | Відсутній стабільний приріст проти baseline |
| `R(t)` зменшує імпульсивні помилки | Predictive control + verification friction | False-Accept Rate, overrule-rate | FAR не знижується або росте в A/B |
| Стрес-контури можна виявляти до помилки | PFC stress vulnerability | lagged AUC/PR для error burst | EEG/HRV модель не краща за поведінковий baseline |
| Longitudinal тренування дає transfer | Adaptive resilience loop | latency↓ при accuracy↔/↑ | latency покращується лише ціною accuracy↓ |

---

## 5) Операційний стандарт для документації та коду

1. **Правило подвійної опори:** кожен новий теоретичний claim посилається мінімум на 1 S-tier + 1 A/B-tier джерело.
2. **Правило методичної прозорості:** кожна нова метрика (`PLV`, `HRV`, `Hurst`, `MFDFA`) має citation до первинної методології.
3. **Правило статистичної гігієни:** preregistration, power estimate, multiple-comparison control — обовʼязкові.
4. **Правило інженерної трасованості:** docs-claim має бути мапований на конкретний модуль/інваріант або тест.

---

## 6) Формат посилань у майбутніх оновленнях

Рекомендований шаблон для нових джерел:
- `Author(s). (Year). Title. Journal/Publisher, volume(issue), pages. DOI`
- Обовʼязкові поля: **тип доказу**, **межі валідності**, **що саме в системі цим обґрунтовується**.


## 7) Рівні доказовості (для всіх нових claim у docs)

- **Established:** підтверджено консенсусними оглядами/реплікаціями, низька концептуальна невизначеність.
- **Strongly Plausible:** сильний механізм + узгоджені дані, але є контекстні межі генералізації.
- **Tentative:** робоча гіпотеза з частковими даними, потребує preregistered перевірки.
- **Unsupported/Weak:** популярна теза без достатньої емпіричної опори.

**Обовʼязкове правило:** у docs поруч із кожним нетривіальним твердженням вказувати рівень доказовості.

---

## 8) Traceability до коду (docs → modules/tests)

| Теоретичний вузол | Модуль у репозиторії | Тестове покриття (приклади) | Мінімальна метрика якості |
|---|---|---|---|
| Phase alignment / PLV | `neurophase/metrics/plv.py` | `tests/test_plv.py` | стабільність оцінки по вікнах |
| Executive gate / `R(t)` | `neurophase/gate/execution_gate.py` | `tests/test_execution_gate.py` | зниження False-Accept Rate |
| Emergent phase dynamics | `neurophase/gate/emergent_phase.py` | `tests/test_emergent_phase.py` | early-warning перед error burst |
| HRV + resilience proxy | `neurophase/risk/*`, `neurophase/state/executive_monitor.py` | `tests/test_evt.py`, `tests/test_executive_monitor.py` | latency↓ без accuracy↓ |
| Complexity/fractal features | `neurophase/risk/mfdfa.py`, `neurophase/metrics/hurst.py` | `tests/test_mfdfa.py`, `tests/test_hurst.py` | out-of-sample stability |

Ця таблиця фіксує, що бібліографія не ізольована від продукту: кожен теоретичний claim має шлях до коду і перевірки.



## 9) Integration execution link

Практичне виконання цієї бібліографічної політики в CI/CD та перед релізом: `docs/validation/integration_readiness_protocol.md`.


## 10) Companion canonical set

Розширена canonical-версія з повним 24-source контуром, DOI/traceability і release-gate чеклістом: `docs/theory/neurophase_elite_bibliography.md`.
