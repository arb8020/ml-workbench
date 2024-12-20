## computation/thinking layers

### multi head attention

when we write or speak, we weigh different parts of the sentence so far to figure out what to say next. 
as an example: if we were continuing the sentence: 'the boy is eating a blue'
we might naively use a word that follows blue frequently, like 'color' - but this doesn't make any sense!
reading the sentence, it seems clear that the next word should be something a boy might eat, that happens to be blue
so we're paying extra attention to the words 'boy' 'eating' and 'blue' to make our prediction
and as we choose the next token, those tokens get weighted more than others, allowing us to pick a word like 'lollipop'

implementation wise, we're using the most recent token, and looking at how related it might be to previous tokens in the sequence
at the end, we want some kind of 'score' of how well the most recent token matches up with previous tokens in the sequence
we can think of this kind of like a fuzzy hashmap
our model has been given the space to learn about the word 'blue', and has some internal representation of it, specifically as a word we're looking to match to
the model has also been given the space to learn about the words 'boy', 'eating', etc, and has an internal representation of it, specifically as a word to match with
and the model also has been given the space to learn about all the tokens, creating internal representations specifically as a word that's been matched
so when we're trying to figure out which tokens to pay attention to
in our little learned fuzzy hashmap, the model might get a match value between 'blue' and 'boy' of like 0.6, and apply that similarity score to the 'boy' value
and we'll end up with some vector of attention values for the whole sequence, maybe like [3, 14, 2, 15, 4, 11]
applying softmax, we get a probability distribution, in this case used as a weighting mechanism for each of the previous tokens
and now our model can use that weighting to make better predictions about that next token
interestingly, we usually also do this in parallel, so we can learn not just semantic relations but also grammatical, parts of speech, etc

rigorously, we give the model a matrix for queries (how to match to this token), keys (how to match with this token), and values (what this token 'means')
these matrices are shape (embed_dim, embed_dim) -> from the model's initial embedding dimension representation of the raw token, to a new space where it can represent the information it wants relating to attention
while running the forward pass, we take the query token corresponding to the last token in our sequence, and get its cosine similarity with the key vectors of the other tokens in the sequence (including itself)
we do this with a dot product, as the cosine similarity is basically just a scaled dot product
this gets scaled by the square root of the (embed_dim) because we don't want these unscaled dot products to get too big
when training, we mask out the values of the sentence the model isn't supposed to have seen yet - but this doesn't apply to inference
as a quick note on masking, we basically create a lower triangular matrix of really large negative values and add it to our matrix of scaled dot products
now, when we perform softmax to turn attention values into the weighting for how much attention to pay to each token
the large negative values effectively zero out the tokens we shouldn't know about - exactly the behavior we want!
that weighting gets multiplied with the values of each token, which is the final part of the model's core 'prediction' for the next token
and finally projected back into the actual token embedding representation of the model, to get token logits for the output
to add in the parallel computation of different types of attention scores at the same time, we usually split the embedding dimension into an n_head parameter. n_head=2 for two types of relationships, etc
note that we don't add new parameters! the size of each head's embedding dimension (head_dim) is a divisor of the initial embed_dim parameter (head_dim = embed_dim//n_head)
we create these heads simply by splitting up the initial (embed_dim, embed_dim) matrix, and concatenating the results later
so the top half might end up being the syntactic part of attention, the bottom half might end up being the semantic part of attention, etc

### feedforward/fully connected network

while the attention mechanism is powerful, giving us the context of what tokens to pay attention to
we currently still just have a fancy n-gram model, we haven't added anything beyond adds/multiplications
the feedforward network is a foundational building block of many deep learning architectures
more discussion on this can be found in the activation function section, but roughly the nonlinear functions allow us to approximate a much larger class of functions than affine transformations alone
specifically, we project the embedding dimension into a higher dimensional thinking dimension, usually 4x the embedding dimension (powers of 2 convenient for GPUs + enough space to be meaningful)
then, we apply the activation functions, learn complex relations, and then compress back down into the embed_dim, keeping only the useful stuff
so now we have a neat little cycle we apply over and over again
learn relations between tokens -> learn about each token's meaning
so we combine both cross-token interaction, and the power of deep learning

## activation functions

what really is an activation function?
we came up with them when we realized that the almighty 'draw a line through your data' couldn't solve everything
for example, consider a drug that has a dosage-response relationship that is low in both an under/overdose scenario, but high in the right dosage
your silly little 'linear regression', however powerful it may be, literally cannot learn this relation
so someone came with with the idea of adding some kind of nonlinearity, that we could compose and transform into an arbitrary shape
for example, imagine a call option payo- i mean the function max(0,x)
if we transformed and shifted a bunch of these together, we could theoretically fit any polynomial
this is known as the 'universal function approximation theorem'
the activation function is that max(0,x) nonlinear function that we compose together - its also known as ReLU (rectified linear unit) (don't ask me why)
we also have the activation function gelu in this codebase
and i wish i had a cool intuitive reason for why its used, but the people who wrote the GPT-2 paper literally just said it worked better empirically
could expand on this in the future

## regularlization and normalization

#### layer norm

the repeated application of the attention and feedforward layers doesn't give the model a chance to 'recalibrate' so to speak, in between processing steps
mathematically, we might be getting some weirdly skewed or shifted values from one layer, and we propagate that skew or whatever it is immediately
so we just take our distribution from the most recent layer, and we normalize it into mean 0 and variance 1
we also allow the model to take an affine transformation (scale/shift) on that distribution, in case something different works better for the model
this has been found to smooth out the loss surface and make it easier for the model to learn

### residual connections

you may have noticed these models are quite deep and large
one might wonder, why not just make these models deeper and larger? 
more opportunities for hierarchical representation, and more space to think is good right? 
well back in the day, computer vision researchers had the same idea as you
shocker, it is not so easy
what happened was that the researchers were trying to train really deep models
but found that it didn't strictly improve model performance - which is kind of weird if you think about it
suppose we have a model that is some n layers deep
we would assume that a model n+1 layers deep should always be at least as good as the n layer model
after all, it could just replicate the n layer deep function
and use the identity function in any excess layers
but empirically, training deeper networks actually had higher training loss

roughly, as we go through these complex transformations
our initial weights early in the model can't really easily learn from/access the gradient from the input we just used
its hard for the model to preserve both the complex transformations its learning, as well as the original data
imagine playing a game of telephone, where the first person asks a question about the group of people
and the last person has to answer the question
first of all, its going to be tough to keep the original question correct, but we also need to add info to each transmission
but it would be so much easier to skip ahead and remind someone halfway through the line what the original question was 
or better yet, tell everyone about it

rigorously, we're trying to estimate some function H(x)
and we're doing it by finding some intermediate functions a(b(c(d(x)))) = H(x)
since each of those functions is nonlinear, we can't really preserve x, if one of these subfunctions needs to be f(x) = x
and it turns out that one of the functions learning it should exactly preserve an input is kinda hard
but luckily, its pretty easy for functions to learn to just not do anything at all
so we should try to have each function predict what change it needs to make to x in a more pure manner
we reformulate the problem as trying to predict some H(x) = F(x) + x
since each intermediate function predicts the difference to apply to x, and learning F(x) = 0 is easy
this makes training a deep model way smoother

we call this trick a 'residual connection'

## loss

### softmax

next token prediction is basically a classification task
suppose we had to pick between [red, green, blue] in some computer vision model
the correct 'label' might be [1, 0, 0]
a classification model usually compares the label to some probability distribution output
as an example, our model output might be one of [0.9, 0.02, 0.08] or [0.4, 0.3, 0.3]
while the maximum value for both is correct
its clear that one model potentially performs better at the task
as its probability distribution is in some sense 'closer' to the true distribution

but we don't have probability distributions by default
our model might output raw logits like [5, 3, 2]
we need to turn this into a probability

call our initial output logits f(x)
we need to create probability distribution p(x), while losing minimal information from f(x)
the best representation of some knowledge state is the probability distribution that maximizes entropy
entropy (in the information theory context) is the expected value of surprise
surprise, as it turns out, is actually a mathematical quantity
to develop intuition about this, suppose you had 100 red balls and 2 blue balls in a box
drawing a blue ball out is more 'surprising' than a red ball
so we might think the surprise might be 1/p
but if p is 0 this breaks, so we just add a log transform
thus, the rigorous probability definition of surprise is log(1/p)
and entropy is E = sum (p(x) * log(1/p(x)))
this can be simplified into E = -sum(p(x) * log(p(x)))

so we can set up a constrained optimization problem
to figure out a function that can map f(x) into the best p(x)
our objective function is maximizing E = -sum(p(x) * log(p(x)))
our constraints are that
1. sum(p(x)) = 1
2. p_i >= 0

using lagrange multipliers (not shown here, too long), we can solve the above to get
p_i = e^{beta * x_i}/sum(e^{beta * x_j})
for writing the softmax, we'll remove beta
but remember that we can scale the logits before doing the softmax
this beta constant is frequently rewritten as 1/T, where T is temperature
we call it that because of the boltzmann distribution in statistical mechanics, which is out of scope

you might have noticed we have a bit of instability with this computation
e^x where x is a really big number can easily overflow
but e^-x will always be between 0 and 1
so we want to see if we can shift all of our e^x such that x is, at most, 0
suppose we subtracted some constant c from all of our x_i (from the initial f(x))
mathematically, our function is invariant to any shifts (can you prove this)
so we can just subtract the largest value from every value
to get a numerically stable softmax

### cross entropy loss

in order to actually calculate the loss function between some true p(x) and our predicted q(x)
we basically take that entropy function 
E = -sum(p(x) * log(p(x)))
and we just replace one of the p(x) distributions with our model output distribution q(x)
C = -sum(p(x) * log(q(x)))
this is known as 'cross-entropy', as its the entropy 'across' two distributions
note that we choose to put the log transform on our q distribution bc log(0) is not stable

## optimizers

### stochastic gradient descent

let's use a 2 parameter example: we have a parameter x and y, each corresponding to an axis
and the loss surface is some value z for each of these parameters x,y
imagine a blind person on this loss surface: they can tell which way, locally, is the correct next step to take by feeling the slope under their feet
and at each step, they get progressively closer to some minimum point on the full loss surface
the size of the steps is determined by the learning rate of the function 
but mathmatically we're literally just updating each parameter by the learning rate multiplied by the gradient with respect to the training batch 
there's a little bit of variance at each step, as each step is some random 'batch' of the full dataset, but it's proven to converge for a correct choice of learning rate

### stochastic gradient descent with momentum

stochastic gradient descent works, but is a little slow
the man on the loss surface may be taking multiple steps in the same direction, without taking advantage of the most recent gradients that he felt out
compare this with a bowling ball. the bowling ball carries inertia and momentum from the previous movements made, and reaches the surface's minima faster as a result
so we add in a exponential moving average of the previous gradients to add to each of our updates

### adam

in our examples so far, we've covered the optimization of a 2 parameter model
this is less obvious in our example, but it's actually not clear that each parameter should get an equivalent learning rate
why not make it easier to move north/south than east/west if we need to?
if the surface is very steep to the left, but not steep in front of us, we probably want to take an even bigger step left instead of forward, even after adjusting for the gradient
so we can keep track of the average gradient we've seen so far for a given parameter, as well as the average squared gradient, to get more information on how to update
while, as before, parameters with a high average get bigger steps to keep using momentum
parameters with high variance/squared mean should probably get smaller updates, since its not really clear that the terrain is smooth/stable
as a side note, the initial values of mean/squared mean will be 0 when initialized, so we add a bias correction mechanism that incorporates time step to account for this

## tokenization

our model unfortunately does not have the ability to take in raw strings, as words are not math
you may think 'oh lets just take the ascii/unicode value of each character' but there are some problems with this
the most clear framing is the context length of the model
at longer context lengths, inference would take longer, due to having to generate more characters, but also because attention is a O(seq_len^2) operation
so we maybe don't want each character to be a token
but we also might not want to make each word into a token, as we lose the ability to understand sub-words, and it makes it harder for the model to come up with new words
taking a step back, we're looking at a data compression problem
how can we figure out the representation of our training text/possible training texts that maximally retains information while also decreasing size? 

lets think of each character as a byte
inspecting the data, we might find some bytes follow other bytes very often
so we can maybe just predict both of these at the same time, instead of one and then the other after
let's tag this pair of bytes with a new value, say 256 (after 255)
now, we can go back through our data and find the next most common pair of bytes, and compress them into another new value, say 257
repeatedly applying this process until we get to some desired vocab_size, or sufficient compression, we can efficiently represent some training data
this is known as byte pair encoding. the current codebase implements a rudimentary/slow version of it, but theres a lot of room for optimization in the future
