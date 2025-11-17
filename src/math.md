
# Neural Networks From Scratch

Nous détaillons dans ce document la partie mathématique.

## Définitions

Un réseau de neurones est donné par 

$f(w, x) = g_1(w_1, g_2(w_2, g_3(w_3, ...)))$

où 
- $w$ désigne les paramètres du réseau (poids et biais)
- $w_k$ peut être une matrice, ou être aussi un tuple contenant des poids un vecteur de biais. Donc $w$ est effectivement un vecteur dont chaque composante peut être une matrice ou un tuple contenant une matrice et un vecteurs de biais. Ces matrices peuvent avoir des dimensions différentes d'un indice à l'autre.
- $x$ désigne l'input du réseau (un vecteur ou une image ou autre...)

Dériver la fonction avec les points de suspension pouvant être une source d'erreurs, on utilise cette définition récursive à la place, où $g_1$ désigne la dernière couche du réseau et $r_2$ désigne le reste des couches du réseau.

$f(w,x)= g_1\Big(w_1, r_2(w_{2:}, x)\Big)$

où

$w_{k:}$ désigne $(w_k, w_{k+1}, w_{k+2},..., w_N)$ pour utiliser une notation analogue à celle utilisée dans les bibliothèques Numpy et Pytorch.

Et le reste du réseau $r_2$ est calculé de la sorte:

$r_2(w_{2:}, x) = g_2(w_2, r_3(w_{3:}, x))$

## Calcul de la dérivée

Pour étudier les variations on choisit $d = (d_1, d_2, ...)$ petit

Le but étant de nous conduire à un développement limité d'ordre 1 comme suit
 
$f\Big(w + d,x\Big) \approx f\Big(w, x\Big) + \frac{\partial f}{\partial w}(w) . d$

En remplaçant $f$ par son expression récursive et en développant on obtient

$f\Big(w + d,x\Big)$

$= g_1\Big(w_1 + d_1, r_2(w_{2:} + d_{2:}, x)\Big)$

$\approx g_1\Big(w_1, r_2(w_{2:}, x) + \frac{\partial r_2}{\partial w_{2:}}(w_{2:}, x) d_{2:}\Big) + \frac{\partial g_1}{\partial w_1}\Big(w_1, r_2(w_{2:}, x) \Big) d_1$


$\approx g_1\Big(w_1, r_2(w_{2:}, x) \Big) + \frac{\partial g_1}{\partial r_2}(w_1, r_2(w_{2:}, x)) \frac{\partial r_2}{\partial w_{2:}}(w_{2:}, x)d_{2:} + \frac{\partial g_1}{\partial w_1}\Big(w_1, r_2(w_{2:}, x) \Big) d_1$

Nous pouvons à présent facilement identifier la dérivée de $f$ comme étant les termes qui multiplient les composantes de $d$.

## Simplifications du calcul de la dérivée

En observant cette dernière formule, nous pouvons constater que la dérivée d'une couche $k$ nécessite la valeur de $r_k(w_{k}, x)$ deux fois. Mais à la place de la recalculer nous pouvons simplement la garder en mémoire après l'avoir calculée lors de la passe *forward*.

A priori on pourrait penser que ceci nous amènerait à la calculer une fois à la place de trois (1 forward + 2 backward). Mais, si l'on ne gardait pas la valeur en mémoire et que l'on la recalculait à chaque fois, le calcul de $r_3$, nécessiterait lui aussi le calcul de $r_k$ ($k >> 5)$, tout comme le calcul de $r_4$ et le calcul de $r_5$...

Le gain est donc bien plus important.

**TODO**: faire un calcul de gain de complexité

**TODO**: Un autre constat est que le calcul la dérivée du reste $\frac{\partial r_2}{\partial w_{2:}}$ gagnerait à ressembler au calcul de la valeur.

**TODO** valider ceci avec des citations et des références discrètes (liens) aux sources.



