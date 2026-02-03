#import "@preview/diatypst:0.8.0": *
#import "@preview/polylux:0.4.0": *

// TODO Dag
// TODO bases de fonctions
// https://docs.pytorch.org/docs/stable/notes/autograd.html
// Autograd mechanics — PyTorch 2.9 documentation
// https://wangkuiyi.github.io/jacobian.html
// https://pytensor.readthedocs.io/en/latest/gallery/autodiff/vector_jacobian_product.html

// Disicpliné
// TODO mesurer combien de temps pour entraîner un réseau avec mon mac
// 
// Voir la partie de Hakan et comment merge


// Partie math 1er
// VJP
// Perceptron + MLP + CNN
// 
// a : activation
// z : combinaison linéaire
// x0 = x[0] ou x0 ? : entrée
// y = sortie finale
// 


#show: slides.with(
  title: "Neural Networks from Scratch", // Required
  subtitle: "Elements de théorie",
  date: "01.01.2026",
  authors: (""),

  // Optional Styling (for more and explanation of options take a look at the typst universe)
  ratio: 16/9,
  layout: "medium",
  title-color: blue.darken(60%),
  toc: true,
)



// == Plan

// - Réseau de neurones comme MLP
// - Dérivée
// - Calcul automatique de la dérivée
// - Implémentation de classes Python pour la dérivée symbolique
// - Formulation d'un problème d'optimisation
// - Optimisation
// - Utilisation de plusieurs plusieurs couches
// - Constat : prend du temps
// - Backpropagation





== Objectifs
- S'amuser
- Uniquement les aspects théoriques des réseaux de neurones (uniquement Python, Numpy)
- Implémenter un autodiff
// - Implémenter la backpropagation
// - Implémenter un réseau convolutif pour la classification d'images
- Incrémental
- Comprendre l'apport d'une librairie de réseaux de neurones


= Réseau de neurones
== Qu'est ce qu'un réseau de neurones ?


// TODO intégrer partie Hakan

#set align(center)
#image("images/dag_color.webp")

== Dérivée
#slide[
/ *Question*: A quoi sert la dérivée d'une fonction pour l'entrainement d'un réseau de neurones ?

#uncover((beginning:2))[
- Donner la pente au voisinnage d'un point
//  ]
// #uncover((beginning:2))[
  $ f'(x) = lim_(h -> 0) (f(x + h) - f(x))/(h) $
//  ]

// #uncover((beginning:3))[
- Donner une approximation locale de la fonction
//  ]
// #uncover((beginning:3))[
  $ f(x + h) approx f(x) + f'(x).h $
//  ]

//#uncover((beginning:4))[
- Permettre de minimiser une fonction de coût
//  ]
//#uncover((beginning:4))[
  $ x_(n+1) = x_n - eta . f'(x_n) $
]

]
= Autodiff - calcul automatique de la dérivée
== Formules usuelles pour la dérivée de fonctions de $bb(R)$ dans $bb(R)$

// table containiing the different rules
#set align(center)
#set align(horizon)

#table(
  columns: 3,
  table.header(
    [Nom de la règle],
    [Formule],
    [Dérivée]
  ),
  [Constante],[$C$], [$0$],
  [Carré],[$x^2$], [$2 x$],
  [Somme],[$f(x) + g(x)$], [$f'(x) + g'(x)$],
  [Produit],[$f(x) g(x)$], [$f'(x) g(x) + f(x) g'(x)$],
  [Composition],[$f(g(x))$], [$f'(g(x)).g'(x)$],
  [Sigmoide],[$sigma(x) = 1 / (1 + e^(-x))$], [$sigma(x)(1 - sigma(x))$]   

)

== Exercice 01.a : Implémentation en Python

#set align(left)

Sur le modèle ci-dessous, implémentez :
- la dérivée d'une multiplication
- la dérivée de $x->x^2$


```python
class Add(Op):
    def __init__(self, left: Op, right: Op):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} + {self.right})"

    def diff(self):
        return Add(self.left.diff(), self.right.diff())

    def eval(self):
        return self.left.eval() + self.right.eval()
```



== Exercice 01.b : Test de l'autodiff
#set align(horizon)
- Avec votre code Python, calculez la dérivée de l'expression suivante : $(x + 2) * (x + 3) $

- Vérifiez qu'elle correspond bien à 

$ (x + 3) + (x + 2) = 2 x + 5 $

- Quid de l'expression $(x + 1) * (y + 2)$ ? 


== Exercice 01.c : estimation d'un paramètre

#set align(horizon + left)
Estimer la moyenne $mu$ d'un jeu de données $X_1,...,X_N$ où 
- les $X_i$ sont iid
- et sont tirées selon $cal(N)(mu, 1)$

*Objectif* : minimiser une fonction de coût $cal(L)$ 

$ cal(L)(hat(mu)) = sum_i (X_i - hat(mu))^2 $

Vérifier que $hat(mu) approx mu$



== Quelles sont les améliorations possibles dans le code ?
// TODO uncover
- Enlever automatiquement les 0 des additions
- Simplifier automatiquement les multiplications par 1 
- Surcharger les opérateurs Python (+, \*, ...)
- Afficher le graphe de calcul (le fameux DAG)
- Permettre de dériver par rapport à plusieurs variables


= Dérivée de fonctions de plusieurs variables
== Derivee multi-variables

#set align(left + horizon)
#slide[
Pour calculer une approximation locale de la fonction $x,y mapsto f(x,y)$

#set align(horizon)
$ f(x + epsilon_x, y + epsilon_y)  approx f(x, y) + (partial f)/(partial x) epsilon_x + (partial f)/(partial y) epsilon_y $


//#uncover((beginning:2))[
Similaire à $ f(x + epsilon) approx f(x) + f'(x).epsilon $
//]

]

== Exercice : Implémentation de la dérivée multi-variables

#slide[


/ *Question* : Comment modifier le code existant pour implémenter la dérivée multi-variables ?



#uncover((beginning:2))[
Une solution, dans la classe `Variable` :
#image("images/diff_wrt.png")
]

]


== Exercice : optimisation

#set align(horizon + left)
On souhaite ajuster une droite $y = a x + b$ sur un jeu de données.



*Objectif* : Etant les données générées (voir exercice), minimiser une fonction de coût $cal(L)$ 

$ cal(L)(a,b) = sum_i (y_i - (a x_i + b))^2 $



== Gradient et Jacobienne
#slide[

On souhaite calculer une approximation locale de la fonction $x -> f(x)$ *où $x$ est un vecteur* et contient de nombreuses variables. 

#set align(horizon)
Lorsque $f : bb(R)^n -> bb(R)^m$, $J f(x)$ est une matrice
$ f(x + epsilon)  approx f(x) + J f(x). epsilon $

Lorsque $f : bb(R)^n -> bb(R)$, $nabla f(x)$ est un vecteur de $bb(R)^n$
$ f(x + epsilon)  approx f(x) + nabla f(x). epsilon $


#uncover((beginning:2))[
Similaire à lorsque $f : bb(R) -> bb(R)$ 
$ f(x + epsilon) approx f(x) + f'(x).epsilon $
]

]
= Dérivée directionnelle

== Dérivée directionnelle
#slide[

$f: bb(R)^n times bb(R)^m  -> bb(R)^k$ est une fonction de deux vecteurs :

$ x, y mapsto f(x,y) $

On définit sa dérivée directionnelle par rapport à $x$ ainsi :

$ (partial f)/(partial x)(x, y).v = lim_(h->0)(f(x + h v) - f(x))/h $

#uncover((beginning:2))[On utilisera aussi la notation plus légère
$ (partial f)/(partial x)(x, y).epsilon = partial_x f(x, y).epsilon $
]
]
== TODO
/ *Intérêt* : dérivée couche par couche

== Formules usuelles

- La dérivée de la composée de fonctions et des applications linéaires se marie bien dans le cadre des réseaux de neurones.\

$ f(x + epsilon) approx f(x) + partial f(x).epsilon $
#set align(center)
#set align(horizon)

#table(
  columns: 3,
  table.header(
    [Nom de la règle],
    [Formule],
    [Dérivée]
  ),
  [Constante],[$C$], [$0$],
  [Application linéaire], [$A.x$], [$A.epsilon$],
  [Application bilinéaire], [$B(x_1, x_2)$], [$B(x_1, epsilon_2) + B(epsilon_1, x_2)$],
  [Norme au carré],[$||x||^2$], [$2 x.epsilon$],
  [Somme],[$f(x) + g(x)$], [$partial f(x) epsilon + partial g(x) epsilon$],
  [Produit (matriciel)],[$f(x).g(x)$], [$(partial f(x).epsilon).g(x) + f(x). partial g(x).epsilon$],
  [Composition],[$f(g(x))$], [$partial f(g(x)). partial g(x).epsilon$],
  
)
#set align(left)
\*La non-commutativité doit être respectée pour les produits.



// == Différentiation : récapitulatif

// Approximations locales. Quels sont les avantages/inconvénients de chacune ?

// - La dérivée d'une fonction d'une variable réelle :
// $ f(x + h) approx f(x) + f'(x).h $
// - (2) La dérivée de plusieurs variables réelles :
// $ f(x+ epsilon_x ,y + epsilon_y) approx f(x, y) + (partial f) / (partial x)(x,y). epsilon_x + (partial f) / (partial y)(x,y). epsilon_y $
// - La dérivée par rapport à un vecteur :
// $ f(x + epsilon) approx f(x) + nabla f(x).epsilon $
// - La dérivée directionnelle, par rapport à plusieurs vecteurs (même formule que (2), mais avec des vecteurs).

// //$ f(x + Delta_x, y + Delta_y) approx f(x, y) + partial_x f(x,y).Delta_x + partial_y f(x,y).Delta_y $

==
/ *Question* : Quels sont les avantages et inconvénients de chacune des formules ?

- Dérivée directionnelle : permet de rester au niveau d'abstraction souhaitée. Manier plusieurs vecteurs/matrices.

= Backpropagation

== Constat sur la répétition du même calcul
#slide[
Réseau de neurones : empilement de couches 

$ y(w, x) = f_n (...f_3(w_3, f_2(w_2, f_1(w_1, f_0(w_0, x))))...) $

Ou en définissant le réseau de manière récursive

$ y(w, x) = y_n (w_(0:n), x) = f_n (w_n, y_(n-1)(w_(0:n-1), x)) $

]


#let colblue(x) = text(fill : rgb("#0000FF"), $bold(#x)$)
#let colgreen(x) = text(fill : rgb("#008000"), $bold(#x)$)
#let colred(x) = text(fill : rgb("#FF0000"), $bold(#x)$)


== Dérivée du réseau de neurones
#slide[
  On dérive par rapport aux paramètres d'une couche $k < n$ :

$ colred((partial y_n)/(partial w_k)) &= (partial  f_n)/(partial y) . colred((partial y_(n-1))/(partial w_k)) $



/ *Question*: Que remarque t'on ?



#uncover(2)[
Une formule récursive apparait
// TODO check indices
$ ((partial y_n)/(partial w_k)) &=   (product_(i=k+1)^(n) (partial f_(i))/(partial y_(i))) (partial f_k)/(partial w_k) $

  

  $=>$ On aimerait mémoriser _(memoize)_ le résultat pour le réutiliser.\
  $=>$ La dérivée de la fonction et la fonction gagneraient à se ressembler.
  ]
]

== Comment calculer le gradient ?

#slide[
$ colred(partial f(...). partial g(...)) . Delta w $
/ *Question*: On souhaite calculer le terme qui multiplie $Delta w$, qui est le gradient.\
  *Problème* : les opérations que l'on a défini (FeedForward, Convolution) sont définies pour prendre une entrée à droite et donner une sortie 
  $ y = A . x $

*Comment faire ?*\
#uncover((beginning:2))[
Remplacer $Delta w$ par chacun de vecteurs de la base canonique ?\ 
]
#uncover((beginning:3))[
Constater que les opérations sont toutes linéaires, et que par associativité


]

]


