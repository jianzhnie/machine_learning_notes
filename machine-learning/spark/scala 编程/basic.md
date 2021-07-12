## **1. Variables**

We can define immutable variables using `val`:

```scala
scala> val name = "King"
name: String = King
```

Mutable variables can be defined and modified using `var`:

```scala
scala> var name = "King"
name: String = Kingscala> name = "Arthur"
name: String = Arthur
```

We use `def` to assign a label to an immutable value whose evaluation is deferred for a later time. It means the label’s value is lazily evaluated every time upon use.

```scala
scala> var name = "King"
name: String = Kingscala> def alias = name
alias: Stringscala> alias
res2: String = King
```
While defining `alias`, no value was assigned to `alias: String` since it is lazily associated, when we invoke it. What would happen if we change the value of `name`?

```scala
scala> alias
res5: String = Kingscala> name = "Arthur, King Arthur"
name: String = Arthur, King Arthurscala> alias
res6: String = Arthur, King Arthur
```



## 2. Control flow

We use control flow statements to express our decision logic.

You can write an `if-else` statement as below:

```scala
if(name.contains("Arthur")) {
  print("Entombed sword")
} else {
  print("You're not entitled to this sword")
}
```

Or, you can use `while`:

```scala
var attempts = 0
while (attempts < 3) {
  drawSword()
  attempts += 1
}
```



## 3. Collections

Scala explicitly distinguishes between immutable versus mutable collections — right from the package namespace itself ( `scala.collection.immutable` or `scala.collection.mutable`).

Unlike immutable collections, mutable collections can be updated or extended in place. This enables us to change, add, or remove elements as a side effect.

But performing addition, removal, or update operations on immutable collections returns a new collection instead.

Immutable collections are always automatically imported via the `scala._ `(which also contains alias for `scala.collection.immutable.List`).

However, to use mutable collections, you need to explicitly import `scala.collection.mutable.List`.

In the spirit of functional programming, we’ll primarily base our examples on immutable aspects of the language, with minor detours into the mutable side.



### **List**

We can create a list in various ways:

```scala
scala> val names = List("Arthur", "Uther", "Mordred", "Vortigern")names: List[String] = List(Arthur, Uther, Mordred, Vortigern)
```

Another handy approach is to define a list using the cons `::` operator. This joins a head element with the remaining tail of a list.

```scala
scala> val name = "Arthur" :: "Uther" :: "Mordred" :: "Vortigern" :: Nilname: List[String] = List(Arthur, Uther, Mordred, Vortigern)
```

Which is equivalent to:

```scala
scala> val name = "Arthur" :: ("Uther" :: ("Mordred" :: ("Vortigern" :: Nil)))name: List[String] = List(Arthur, Uther, Mordred, Vortigern)
```

We can access list elements directly by their index. Remember Scala uses zero-based indexing:

```scala
scala> name(2)res7: String = Mordred
```

Some common helper methods include:

`list.head`, which returns the first element:

```scala
scala> name.headres8: String = Arthur
```

`list.tail`, which returns the tail of a list (which includes everything except the head):

```scala
scala> name.tailres9: List[String] = List(Uther, Mordred, Vortigern)
```

### **Set**

`Set` allows us to create a non-repeated group of entities. `List` doesn’t eliminate duplicates by default.

```scala
scala> val nameswithDuplicates = List("Arthur", "Uther", "Mordred", "Vortigern", "Arthur", "Uther")nameswithDuplicates: List[String] = List(Arthur, Uther, Mordred, Vortigern, Arthur, Uther)
```

Here, ‘Arthur’ is repeated twice, and so is ‘Uther’.

Let’s create a Set with the same names. Notice how it excludes the duplicates.

```scala
scala> val uniqueNames = Set("Arthur", "Uther", "Mordred", "Vortigern", "Arthur", "Uther")uniqueNames: scala.collection.immutable.Set[String] = Set(Arthur, Uther, Mordred, Vortigern)
```

We can check for the existence of specific element in Set using `contains()`:

```scala
scala> uniqueNames.contains("Vortigern")
res0: Boolean = true
```

We can add elements to a Set using the + method (which takes `varargs` i.e. variable-length arguments)

```scala
scala> uniqueNames + ("Igraine", "Elsa", "Guenevere")
res0: scala.collection.immutable.Set[String] = Set(Arthur, Elsa, Vortigern, Guenevere, Mordred, Igraine, Uther)
```

Similarly we can remove elements using the `-` method

```scala
scala> uniqueNames - "Elsa"res1: scala.collection.immutable.Set[String] = Set(Arthur, Uther, Mordred, Vortigern)
```

### **Map**

`Map` is an iterable collection which contains mappings from `key` elements to respective `value` elements, which can be created as:

```scala
scala> val kingSpouses = Map(
 | "King Uther" -> "Igraine",
 | "Vortigern" -> "Elsa",
 | "King Arthur" -> "Guenevere"
 | )kingSpouses: scala.collection.immutable.Map[String,String] = Map(King Uther -> Igraine, Vortigern -> Elsa, King Arthur -> Guenevere)
```

Values for a specific key in map can be accessed as:

```scala
scala> kingSpouses("Vortigern")
res0: String = Elsa
```

We can add an entry to Map using the `+` method:

```scala
scala> kingSpouses + ("Launcelot" -> "Elaine")
res0: scala.collection.immutable.Map[String,String] = Map(King Uther -> Igraine, Vortigern -> Elsa, King Arthur -> Guenevere, Launcelot -> Elaine)
```

To modify an existing mapping, we simply re-add the updated key-value:

```scala
scala> kingSpouses + ("Launcelot" -> "Guenevere")
res1: scala.collection.immutable.Map[String,String] = Map(King Uther -> Igraine, Vortigern -> Elsa, King Arthur -> Guenevere, Launcelot -> Guenevere)
```

Note that since the collection is immutable, each edit operation returns a new collection( `res0`, `res1`) with the changes applied. The original collection `kingSpouses` remains unchanged.

