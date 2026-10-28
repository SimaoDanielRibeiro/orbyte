# DBSCAN

## Intuição

O DBSCAN (*density-based spatial clustering of applications with noise*) vê os grupos como
**zonas densas** de pontos, separadas por zonas vazias. Não é preciso dizer quantos grupos
há, os grupos podem ter qualquer forma (luas, anéis), e os pontos isolados ficam de fora,
como **ruído**.

## Matemática essencial

Com um raio $\varepsilon$ e um número mínimo de pontos $m$:

- um ponto é **núcleo** se tiver pelo menos $m$ pontos (contando com ele) a menos de
  $\varepsilon$;
- dois núcleos a menos de $\varepsilon$ um do outro ficam no mesmo grupo, e os grupos
  crescem de núcleo em núcleo;
- um ponto que não é núcleo, mas está a menos de $\varepsilon$ de um, fica na **fronteira**
  desse grupo;
- os restantes são **ruído** (grupo $-1$).

O DBSCAN do scikit-learn não prevê pontos novos. O laboratório usa uma regra simples: um
ponto novo fica no grupo do núcleo mais próximo, se estiver a menos de $\varepsilon$ dele;
senão, é ruído. Os pontos de treino ficam exatamente com o grupo que o DBSCAN lhes deu.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `eps` | O raio da vizinhança ($\varepsilon$). Pequeno: quase tudo é ruído; grande: tudo se junta num só grupo | 0,5 |
| `min_samples` | Quantos pontos tem de haver no raio para um ponto ser núcleo ($m$). Maior exige zonas mais densas | 5 |

## Quando usar

- Quando os grupos têm **formas irregulares** ou se quer separar o ruído.
- Quando não se sabe quantos grupos há.

## Armadilhas

- **O `eps` depende da escala:** padroniza as features. Para o escolher, ordena a distância de
  cada ponto ao seu `min_samples`-ésimo vizinho e procura o "cotovelo" dessa curva.
- **Densidades diferentes:** um só `eps` não serve a um grupo apertado e a outro espalhado.
  Um raio fica bom para um e mau para o outro.
- **Muitas dimensões:** as distâncias perdem significado, e a densidade também.
- **As métricas de forma** (silhueta, Davies-Bouldin) ignoram o ruído. Olha também para a
  proporção de ruído: mais de metade quer dizer um `eps` pequeno demais.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar e avaliar**, as zonas coloridas são os grupos (com a regra dos
pontos novos) e as cinzentas são o ruído. Experimenta o `eps`: vês os grupos a partirem-se
em bocados (pequeno) ou a fundirem-se num só (grande). Na avaliação, a métrica "Ruído"
diz que fração dos pontos ficou de fora.
