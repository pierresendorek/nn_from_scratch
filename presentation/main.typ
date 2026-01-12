#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof

#show: ams-article.with(
  title: [Neural Networks from Scratch],
  abstract: "Mathematical derivations",
  //bibliography: bibliography("refs.bib"),
)

//#let indicator(A, x) = if calc.includes(A, x) { 1 } else { 0 }

//$ indicator(RR, x) = 1 $

$ "conv"(h, x; s)(i) &= sum_(n) h(n)x(i s + n) bb(1){0 <= n < d_h}bb(1){0 <= i s + (d_h - 1) < d_x} \ 
&= C . x \ 
&= sum_j C(i,j)x(i) $

Lorsque c'est appliqué à $x = delta_l $


$ C . delta_l &= sum C(i,j) delta_l(j) = C(i, l) \
  &= sum_(n) h(n) delta_l (i s + n) bb(1){0 <= n < d_h}bb(1){0 <= i s + (d_h - 1) < d_x} \
  &= sum_(n) h(n) bb(1){l = i s + n} bb(1){0 <= n < d_h}bb(1){0 <= i s + (d_h - 1) < d_x}\
  &= $
  



= Template

Anyone caught using formulas such as $sqrt(x+y)=sqrt(x)+sqrt(y)$
or $1/(x+y) = 1/x + 1/y$ will fail.

The binomial theorem is
$ (x+y)^n=sum_(k=0)^n binom(n, k) x^k y^(n-k). $

A favorite sum of most mathematicians is
$ sum_(n=1)^oo 1/n^2 = pi^2 / 6. $

Likewise a popular integral is
$ integral_(-oo)^oo e^(-x^2) dif x = sqrt(pi) $

#theorem[
  The square of any real number is non-negative.
]

#proof[
  Any real number $x$ satisfies $x > 0$, $x = 0$, or $x < 0$. If $x = 0$,
  then $x^2 = 0 >= 0$. If $x > 0$ then as a positive time a positive is
  positive we have $x^2 = x x > 0$. If $x < 0$ then $−x > 0$ and so by
  what we have just done $x^2 = (−x)^2 > 0$. So in all cases $x^2 ≥ 0$.
]

= Introduction
This is a new section.
You can use tables like @solids.

#figure(
  table(
    columns: (1fr, auto, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [], [*Area*], [*Parameters*]
    ),
    [*Cylinder*],
    $ pi h (D^2 - d^2) / 4 $,
    [$h$: height \
     $D$: outer radius \
     $d$: inner radius],
    [*Tetrahedron*],
    $ sqrt(2) / 12 a^3 $,
    [$a$: edge length]
  ),
  caption: "Solids",
) <solids>

== Things that need to be done
Prove theorems, such as @thm.

#theorem[The Riemann hypothesis is true.] <thm>

#proof[This is left as an exercise to the reader, given the complexity of the theorem.]

= Background
#lorem(40)
