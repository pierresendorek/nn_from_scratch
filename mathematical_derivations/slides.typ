#import "@preview/diatypst:0.8.0": *
#import "@preview/polylux:0.4.0": *

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


== Plan

- Dérivée
- Calcul automatique de la dérivée
- Implémentation de classes Python pour la dérivée symbolique
- Formulation d'un problème d'optimisation
- Optimisation
- Utilisation de plusieurs plusieurs couches
- Constat : prend du temps
- Backpropagation


== Objectifs

- Uniquement les aspects théoriques des réseaux de neurones (uniquement Python, Numpy)
- Implémenter la backpropagation
- Implémenter un réseau convolutif pour la classification d'images
- Incrémental
- Comprendre l'apport d'une librairie de réseaux de neurones


== Dérivée
#slide[
/ *Question*: A quoi sert la dérivée d'une fonction pour l'entrainement d'un réseau de neurones ?

#uncover((beginning:2))[- Donner la pente au voisinnage d'un point]
#uncover((beginning:2))[$ f'(x) = lim_(h -> 0) (f(x + h) - f(x))/(h) $]

#uncover((beginning:3))[- Donner une approximation locale de la fonction]
#uncover((beginning:3))[$ f(x + h) = f(x) + f'(x).h $]

#uncover((beginning:4))[- Permettre de minimiser une fonction de coût]
#uncover((beginning:4))[$ x_(n+1) = x_n - eta . f'(x_n) $]

]

== Calcul automatique de la dérivée

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

== Exercice : Implémentation en Python

#set align(left)

Sur le modèle ci-dessous, implémentez :
- la dérivée d'une multiplication
- la dérivée de $x->x^2$


```python
class Add:
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

#set align(horizon)

==

- Calculez la dérivée de l'expression suivante : $(x + 2) * (x + 3) $

- Vérifiez qu'elle correspond bien à 

$ (x + 3) + (x + 2) = 2 x + 5 $

- Quid de l'expression $(x + 1) * (y + 2)$ ? 




== Quelles sont les améliorations possibles ?

- Enlever automatiquement les 0 des additions
- Simplifier automatiquement les multiplications par 1 
- Surcharger les opérateurs Python (+, \*, ...)
- Permettre de dériver par rapport à plusieurs variables


== Derivee selon plusieurs variables
#set align(left + horizon)
#slide[
On souhaite calculer une approximation locale de la fonction $x,y -> f(x,y)$

#set align(horizon)
$ f(x + Delta x, y + Delta y)  approx f(x, y) + (partial f)/(partial x) Delta x + (partial f)/(partial y) Delta y $



#uncover((beginning:2))[
Similaire à $ f(x + h) = f(x) + f'(x).h $
]

]


== Exercice : optimisation

#set align(horizon + left)
On souhaite ajuster une droite $y = a x + b$ sur un jeu de données.


*Objectif* : minimiser une fonction de coût $J$ 

$ J(a,b) = sum_i (y_i - (a x_i + b))^2 $



== Gradient
#slide[

On souhaite calculer une approximation locale de la fonction $x -> f(x)$ *où $x$ est un vecteur.*

#set align(horizon)
$ f(x + Delta x)  approx f(x) + nabla f(x). Delta x $



#uncover((beginning:2))[
Similaire à $ f(x + h) = f(x) + f'(x).h $
]

]