# LEARNING 1 — repayment log

## Pre Unit -- Learning the Language of the Experiment

### The Experiment in minature
Given an example input: [3, 1, 2]

And Operations:
 reverse: [3, 1, 2] -> [2, 1, 3]
 increment: [3, 1, 2] -> [4, 2, 3]
 sort_asc:  [3, 1, 2] -> [1, 2, 3]

Then ask the system: "First reverse it, then increment it"

Broadly these experiments are asking: Can a neural network learn resuable little operations, then recombine them into compositions it was never trained on?

### Definitions
|Word|Definition  |
|--|--|
|Primitive|A basic operation in the task, like "reverse" or "increment"|
|Composition|Applying the primitives in a sequence. reverse(x) -> increment(x) or increment(reverse(x)|
|Atom|A learned neural model I hope comes to perform one reusable primitive like operation|
|Composer|The learned part of the model that decides which atom to use for a requested operation|
|Facotrization|Solving a larger task by breaking it into reusable pieces rather than learnign each combo seperately|
|Co-apaptation|Modules/Atoms learning to depend on eachothers quirks instead of independently implementing a reusable operation|
|Code|The models internal 512 number vector of the input|
|Manifold|The region or set of internal codes corresponding to meaningful valid respresented inputs (a 512 vector has a lot of space and nonsens in it so the manifold is just the useful part of the vector)|
|Closed Map|The atoms do not rely on the decoder to understand its output but outputs an answer near the manifold without help from decoder|
|Coverage|How many different primitives the learned atoms collectively appear to implement|
|Arm|One experimental condition|
|Split|Which primitive compositions are shown during training vs deliberately held out for testing|

### Atoms - 2 Views
weight-space view: Is the knowledge physically isolated inside a particular block of parameters?

function-space view: Regardless of how the parameters are arranged, does this learned component behave like a reusable function?

### Peak Resident Memory vs FLOPs
FLOPS: How much arithmetic work did the computation perform?

Peak Resident Memory: At the worst moment how much stuff had to be resident in memory simultaneously?

## Unit 0 -- The Question

### Assignment 0.0
**Q1 - What do you expect this document, atoms-preregistration_v2.md, to contain, and what job do you think it serves in the experiment?**
I expect this document to first state the motivation for this experiment/project as well as a general intuition on how to get to the problem driving the motivation. The document likely has a specific broad hypothesis then splits it into a series of falsifyable smaller experiements to build to proving or disproving the overarching hypothesis. As with any good experiemnt/hypothesis they should also list a rough prediction of the otucomes that would suppor tthere hypothesis so that each experiment is falsifyable anf the goal posts dont shift after the fact. The broad hypothesis i expect to be: "Given a parent/teacher model there are small reusable pieces that can be composed together in different configurations to answer the same given input as the parent model without any need of the frozen parent to exist at inference resulting in a sublinear increase in peak resident memeory as the parent model scales."

**Deltas:**
- Preregistration has more careful claim around peak resident memory, explicitly stating fallback  outcome where total atom library might be as large as the parent but only small working set needs to be resident at once
- Preregistraion rather than having broad hypothesis split into smaller series of falsifiable experiemnts it has a core claim then several dependent hypothesises ordered by dependency. Also not all claims need to be falsifiable some can be a fork guiding which direction to go.

Current compression mostly asks, "How can I represent this whole model more cheaply?"
Atoms asks, "How much of what the model knows consists of reusable machinery that many tasks share"

---------------------------------------------------------------------------------------------------------------------
**Q2 - Why does removing the frozen parent make low-rank weight-space atoms much less attractive, while function-space atoms still have a plausible path?**
Removing the frozen parent makes the low rank weight space atoms less attractive because from my understanding something like LoRA which is low rank can only every guide a larger model/matrix in a direction, it doesnt contain anything in and of itself. This is compeltly against the idea and motivation of Atoms. Now im not sure why yet this si the case with low rank in general, I undertsand the adapater part would cause this by whats the relevance with low rank? Now function-space atoms basically remove this criteria/constraint to let Atoms find a way to have a speciality in any way they can.

**Why low rank itself is a bottleneck?**
By definition low rank can only affect a limited sliver of matrix space, so in order to get same full functionality of say a rank 4096 model with rank 16 atoms you end up needing roughly 256 atoms which can make any gained advantage evaporate and this reinvents low rank factorization.

**Learning**
In weight-space:
fixed parent respresentation -> low rank atom must fit it

In function-space:
encoder learns representation <-> atom learns transformation

This idea is the key behind *representational co-design*, which lets the encoder instead learn how to arrange the internal code so that the primitives become transformations these atoms can actually express.

-----------------------------------------------------------------------------------------------------------
**Hypothesis Map**
Ordered by dependency not importance

**H6: Atom Factorization**
- Did we actually learn reusable atoms, or just co-adapted fragments that only work with their usual partners?

**H1: Marginal Compositionality**
- As the Atom library grows, does adding new capability become cheaper because more of it can be built from existing atoms?

**H2: Working-set sparsity**
- Does a working query need only a small fraction of the entire atom library?

**H3: Atom-space geometry**
- What does the structure of the atom library look like, and does that structure suggest storing atoms or generating them?

**H4: Standalone viability**
- Can the Atom system actually operate without the parent model at inference?

**H5: Composer boundedness**
- Does the composer stay small and cheap as the atom library grows, or does the composer gradually become the real model?

----------------------------------------------------------------------------------------------------
**Predict E1_Report**
E1 should test hypothesis 6(H6) first as that is the one that others are dependent on, we first need to know if the system can learn reusable atoms and not just co-adaptive fragments. So the first experiment should first test the simplest version of that, define a set number of primitives that are determinate as well as work with a matrix that the input is encoded into and decoded from. So I would first define the set of primitives and the dimensionality, then I would train the encoder and Atoms to train on a smaller subset of the total combinations of the primitives learning from a small parent model which is frozen and its results. Finally I would run an eval harness that runs thru those same combinations of primitves plus new combinations unseen and see how the accuracy of the encoder/composer + atoms compare to the parent models answers. But one thing is during training the atoms would need to regualy swap partners so to not memorize the output or input of another atom in a sequence but work in any configuration. Also to prove the closed loop is met and the atoms arnt relying on the decoder we would need to measure the accuracy not jsut with result but on coordinates of the manifod it points to.

**Deltas**
- E1 splits into 3 arms so to determine what lever actually causes factorization
	- A1 has fixed co-occurence with no protection against co-adaptation.
	- A2 randomizes co-occurences + atom dropout
	- A3 learns singletons first then freeze the library during pair training
- E1 has no parent model becuase the primitives themselves provide exact ground truth so to not introduce any possible explanations for failure
- E1 runs a A0 case where there is an oracle to tell the correct routing and and intermediate state suprivision to establish a ceiling.

-----------------------------------------------------------------------------------------------------

**Primitives**
8 types of primitives in this experiment:
[`identity`, `reverse`, `increment`, `sort_asc`, `rotate_left`, `swap_halves`, `double`, `reflect`]

- How do you think these primitives will be represented in Python, what inputs/outputs will they operate on, and how do you expect composition of two primitives to be implemented?
	- I think in python the first thing we will want to represent is the function of each primitive, some a function for identity that returns itself, reverse which reverses itself, etc. Then we would want some functions that are called to apply the primitive so that we can also have a function that takes a list of primitves to run sequentially and it can use that function to call primitves one at a time. So input should be a list and output should be a list as all of the primitive logic should operate outside the encoding and decoding.
- Primitives are deterministic python ground truth seperate from the neural network
- Concrete Primitives:
	- identity
		- Do nothing, return input
		- [2, 4, 1, 5] -> [2, 4, 1, 5]
	- reverse
		- reverse the input
		- [2, 4, 1, 5] -> [5, 1, 4, 2]
	- increment
		- Add 1 to every digit
		- [2, 4, 1, 5] -> [3, 5, 2, 6]
	- sort_asc
		- Sort from smallest to largest
		- [2, 4, 1, 5] -> [1, 2, 4, 5]
	- rotate_left
		- Move every position one slot left and wrap first element to end
		- [2, 4, 1, 5] -> [4, 1, 5, 2]
	- swap_halves
		- Sequence is always 8 elements so split into groups of 4 and swap
		- [2, 4, 1, 5, 6, 8, 7, 0] -> [6, 8, 7, 0, 2, 4, 1, 5]
	- double
		- Multiply everything by 2 then modulo 10
		- [2, 4, 1, 5] -> [4, 8, 2, 0]
	- reflect
		- Replace each digit with its refelction so: 9<->0, 1<-> 8, etc
		- [2, 4, 1, 5] -> [7, 5, 8, 4]
- Now v2 replaces sort_asc with index_shift
	- index_shift adds position index to each value and modulo 10
	- [2, 4, 1, 5] -> [2, 5, 3, 8]
- 3 Layers for primitives in python:
	- Define the World
		- functions for what each primitive does
	- Execute Tasks
		- apply primitives
		- apply composition
	- Validate the experimental World, this is important as two different primitive compositions can accidentally compute the same function but more on this later.
		- Check primitive independence
		- Check non commutativity
		- Distinct pair functions

-----------------------------------------------------------------------------------------------------
**Atoms**
- What do I expect Atoms to look like mechanically?
	- So let me start with what I know, the Atom recieves an input that is a array of set length 8 that is encoded into a 512 dimensional vector and it outputs also a 512 dimensional vector that represents a array of length 8 or at least that is the goal. Each Atom starts not knowing what it is as we want to know if it can discover/learn that on its own, so Atoms are most likely a 512? dimensional vector that is multiplied to the input to result in the output. 
	- **Deltas**
		- Atom is not a 512 dimensional vector but rather a small 2 layer neural network represented as 2 512 x 256 matrix of weights plus biases and a nonlinear activation. 
			- 512D input h -> W1: 512X256 -> 256D hidden vector -> GELU -> W2: 256 X512 -> 512D delta -> new state = h + delta
			- W1 maps from state into atoms hidden dimension and w2 maps back to the state dimension
			- GELU is a nonlinear activation function between the atoms two matrix multiplications behaving like a soft gate (large positve values mostly pass thru, values near zero get partially suppressed, and negative values tend to be reduced substantially). The matrices learn what features to combine; GELU gives the atom nonlinearity so it can represent more complicated transformations.
		- Atoms output a change/delta not the output itself, so if h is the input and delta is the output of an atom reverse then we expect new_state = h + delta to be same as enc(reverse(x)) where x is the original list.
	- Residual Application, given current representation what change should I make to it and then add that change to the original state
	- Due to representational co-design an atom never sees the original 8 integers themselves only what the encoder shows them which matters so the encoder cna learn to decide how those 8 integers should be represented in 512D space.

__________________________________________________________________________________
**Composer**
- What do I predict the Composer to look like?
	- First lets define what the composer does, its job is to decide which atoms to use or chain together given an input. So in theory its like a routing system, that means it would need to know the input and if the Atoms represent transformations in hidden state and encoder decides how to encode an input for Atoms then the composer should recieve the un-encoded input so array of 8 integers to then route to the encoder that hands it off to the atoms. This begs the question tho does each atom get an encoding or is the encoder universal for all atoms? Okay so composer recieves the input as 8 integers in an array and also the list of primitives being asked to perform then routes to encoder to atom for each primitive sequentially using the h + delta for each atom result for input into next Atom. Id assume it decides which atom by a score, scoring each one during training on which atom represents which primitive best in a dictionary so at inference it doesnt have to compute that but already has it.
	- **Deltas**
		- One universal shared encoder, so composer recieves the encoded input not the raw input.
		- At each Composer Step the composer recieves 3 things:
			- Current Hidden State
			- Current Requests Primitive
			- Its own memory from previous step
		- Composer does not contain a dictionary like: revers -> Atom 6, etc... but rather every atom owns a learned key vector. Then the Composer produces a query vector and compares with the key in each Atom to get score. The score are routing logits, the routing mechanism then selects an atom using methods such as Grumbel top-1 during training or hard argmax routing in appropriate eval settings.
			- It still computes the routing query and scores at infrence because the composer sees the current state too. This is something to maybe revisit later if we want
	- Composer also does not compute the answer for any one primitive its job is only to choose atom and return the Atom(State) then the delat is gotten and state + delta happens and then next composition step. This is so the composer doesnt secretly become the thing learnign to do it all.
	- Right now we purpusfully let the Composer see the hidden state and primitive because we want to leave room for the possibilit that learned decomposition is stranger than human level labels. But the hidden trap here is that felxibility is exactly where co adaptation can hide. So think of giving the composer state not evidence it needs it but as an affordance
	- Flexible routing allows richer decompositions, but flexibility also creates more ways to fake factorization.

_________________________________________________________________________________
**The conceptual Flow**
x: 8 integer tokens --> SHARED Encoder --> h0: 512 D code --> 

Composer sees: h0 + "reverse" + its memory --> produces query key --> 

compared with atom keys --> one atom selected --> return which atom --> 

Atom(h0) = delta1 --> h1 = h0 + delta1 --> Composer sees: h1 + "increment" + its memory --> 

produces new query key --> compares with atom keys --> one atom selected --> 

return which atom --> Atom(h1) = delta2 --> h2 = h1 + delta2 --> decoder --> 8 output digits


**So that:**
Ground truth world --> defines correct answer
Learned Model --> encoder --> composer --> atoms --> decoder --> tries to discover a solution

------------------------------------------------------------------------------------------------------
**Q3 - Assignment 0.0 - Predictions**
August 11th, 2026
- A1 closed map coverage:
	- Prediction: 1/8
	- Confidence: 85%
	- Reasoning: I say 1/8 because I think identity should be one it can always able to return the input given, there really is no change needed so no real learning needs to happen. As where the others I think will fail because with no anti-co-adaptation then the atoms have no reason to not do the easiest thing and co-adapt with each other as well with fixed co-occurence even if they don't co adapt they will specialize in a way that is still kind of co-adaptation which is to perform one primitive only when preceeded by a known partner which also I think could be partially caused by training the composer with the atoms. If the composer learns that reverse -> increment then it can make the composer memorize patterns rather than learnign any real routing. So I think given all this the A1 experiement will result in 1/8 primitives with an atom independently performing correct tansformation and that primitive is identity because of the reasons ive explained.
- Do you expect replacing `sort_asc` with `index_shift` to cause any **qualitative change** in how the experiment behaves?:
	- Prediction: No qualitative change
	- Which arm(s), if any: None
	- Confidence: 70%
	- Reasoning: I dont expect any qualitative change because I think the anti co-adaptation mechanism is the dominant causal variable so changing one primistives structure should shift difficult/metrics without changing which training procedure factorize
- Do you expect the experiment’s conclusions to depend materially on which primitive combinations happened to be withheld?
	- Prediction: Medium to High Split Sensitivity
	- Confidence: 95%
	- Reasoning: In a ideal perfect factorization then splits shouldnt really matter too much, but given that sometimes different ordered pairs can result into the same underlying function and we see prelim results showing some co-adaptation then I think we can say withholding certian splits can cause different conclusions in the sens of how wrong it is rather than if its wrong or not.
- Which training procedure is most likely to sometimes discover a clean factorization and other times fall into a tangled solution?
	- Prediction: A2
	- Confidence: 70%
	- Reasoning: I say A2 as it is the same as A1 with a new idea, atom drop out and randomized co-occurence and sense I think the random co-occurence does affect how much varience there is then I think this should have the highest variance.

------------------------------------------------------------------------------------------------------------
### Assignment 0.1
**The “could you explain the project to a smart person over lunch?” Test**
-   What problem with today's large models are we trying to solve?
	- Large Models today have a big glaring problem, they can not be ran on consumer hardware even if they are open weights...they simply require too much of the paramers to exist on active memory/RAM.
-   What does the Atoms project suspect is unnecessarily duplicated inside models?
	- The idea driving the Atoms project is that current compression to shrink models is too agnostic of the possibility of shared reusable parts within the model for different kinds of tasks, so Atoms is the idea that there are repeatable reusable machinery that can be composed in different variations.
-   What is the basic alternative Atoms proposes?
	- The alternative Atoms proposes is to not compress the model but rather learn from a frozen parent model what these reusable parts are to then use only what is needed for each query in order to reduce the maximum meemory being activly used on RAM.
-   In ordinary language, what is an atom?
	- An atom in this project is simply put a change that needs to be combined with the input in order to get the expected output that doing a specific task would result in.
-   What does the composer do?
	- The composer in the Atoms Project whole job is to take in the current state and what task needs to be done and simply find which Atom is the best fit for doing that job, nothing else.
-   Why could this reduce peak memory even if the total library remains large?
	- Peak memory could be reduced by doing this even if the library of Atoms remains large because not everything in the library is being used only what is needed.
-   Why must atoms work independently rather than only with familiar partners?
	- An atom must work independently rather than only with familair partners in order to prove that they actually know a specific reusable task as well to be more gnerally composable in different combinatiosn that might be needed for different queries.
-   What does E1 test before attempting the grander claim?
	- E1 is all about testing the idea that a Atom can learn or discover a specific task specialty that is independent and reusable as well as composable.
-   What evidence would make you believe the idea is working?
	- The evidence to lead me to believe the idea is working is firstly that that the result is close or is correct, secondly that before decoding the output that the encoded answer lands in a similiar spot to the actual answer in a 512 dimensional vector, and lastly than unique combinations of unseen primitives/tasks results in a fairly accurate answer as well.
-   **What result would make you conclude the central idea is wrong or not useful?**
	- That new unique unseen combinations of the primitves does not result in any meaningful accuracy on location of the correct answer in the 512 dimensional vector.

**Revisit/Need Work**
-   What is the basic alternative Atoms proposes?
	- The alternative Atoms proposes is to not compress the model but rather learn from a frozen parent model what these reusable parts are to then use only what is needed for each query in order to reduce the maximum meemory being activly used on RAM.
	- **Delta** 
		- learn from frozen parent model is too specific
	- **Adjusted**
		- The alternative Atoms propose is that instead of keeping on giant model resident on memory, we can isntead learn a library of smaller reusable pieces and invoke only the pieces a query needs.
-   In ordinary language, what is an atom?
	- An atom in this project is simply put a change that needs to be combined with the input in order to get the expected output that doing a specific task would result in.
	- **Delta**
		- Atom is not a change but computes the change
	- **Adjusted**
		- An atom is a small learned part of the system that looks at the current internal representation and calculates how to change it to perform some reusable operation.
-   What evidence would make you believe the idea is working?
	- The evidence to lead me to believe the idea is working is firstly that that the result is close or is correct, secondly that before decoding the output that the encoded answer lands in a similiar spot to the actual answer in a 512 dimensional vector, and lastly than unique combinations of unseen primitives/tasks results in a fairly accurate answer as well.
	- **Delta**
		- Dont use the term 512 dimensional vector
	- **Adjusted**
		- The evidence to lead me to believe the idea is working is firstly that that the result is close or is correct, secondly that before decoding the output that the encoded answer lands in a similiar spot to the actual answer in how the system normally represents the correctly transformed input, and lastly than unique combinations of unseen primitives/tasks results in a fairly accurate answer as well.
-   **What result would make you conclude the central idea is wrong or not useful?**
	- That new unique unseen combinations of the primitves does not result in any meaningful accuracy on location of the correct answer in the 512 dimensional vector.
	- **Delta**
		- Not strong enough
	- **Adjusted**
		- The result that would make me conclude the central idea is wrong or not useful is that we can find no resusable primitives in the Atoms independently, or we can but the result is a library where too many of the Atoms need to be used for every query destroying our target of only using some of the atom library.

----------------------------------------------------------------------------
**Feynman explanation**
- “So what exactly is this Atoms experiment, and why should I care?”
	- The Atoms experiment is this idea to solve a problem I see in the world of AI and model today. I mean right now the only way you can use a frontier model is by payign a subscription or investing probably millions or at least 10s of thousands of dollars into building servers if you go the open weights route. That kind of leaves this big disparity that could allow for these big companies to monopolize inteligence itself in a way. So whats the problem for why a typical consumer or at least a small business can't host and run a large open weights model? Something called peak resident memory, somehting like a ssd or harddrive isnt that expensive in all reality but RAM or GPUs are and lots of computers are limited in how much of that they can even have. So my idea is targeted at that gap, which leads me to my hypothesis: A large model consists of enough meaningful duplication that can be extracted, learned, or discovered into smaller reusable pieces that can later be intelegently used on an as needed basis given a specific query. Now this is where the Atom project kind of has multiple moving parts, im calling these reusable pieces Atoms and along with them there needs to be something to decide which atom to use for a given task, Im calling that a composer. Now into my inital first experiment, first thing to do is strip out all the unneeded complexity that coudl itnroduce irrelevant (at this stage) errors. So no parent model, first thing to prove is that Atoms can be learned in general and this can be done by using the same structure of a model which is a matrix with an encoder and decoder. So all that means is the encoder learns to represent a list of 8 numbers in a set size matrix and a decoder does the inverse. Then we use a defined number of primitives, which are just determinate calcuable functions like reverse, increment, etc. Now the whole idea that needs proven is that an Atom can learn one of these primitives thru training it on examples of them like asking reverse then increment etc.  One important caveat here,  after training not only do these Atoms and composer need to get the examples they were trained on correct but also ones they werent trained on and also we can just use the decoded output to compare we need to compare the pre decoded results with the encoded ground truth results we can calculate just by running a function like reverse. Now one thing to also note, is I said the term "train" an encoder, that was very purposeful ehre because it differentiates between two desing choices: weight space and function space. Weight space deltas would be the atoms change or add to the models weights as where function space means the atom itself can focus on representing the transformation rather than how that transformation affects the input. As a consequence to choosing the latter tho we get this idea of representational co-design which allows the encoder to learn how to represent the input in a way that the atom can apply its learned transformation on. Now with all this said what would a convincing postive outcome for E1 look like? Well it would show that the accuracy pre decoding and post decoding is high on unseen combinations of the primitves it was not trained on as well as that only a portion of Atoms that are relevant were used form the Atom library. What would a failure look like? It would look like the Atoms failed to either be accurate on unseen combinations pre decoding or that most of the Atom library was needed to get an accurate result.
	
**Final Feynman repair**
Don't redo the whole speech. Give me just **three cold answers**, 1–2 sentences each:
- Why does H6 have to be established before H1 and H3 mean much?
	- H6 needs to be established before H1 and H3 because if atoms are co-adapted meaning they rely on each other then asking H1, whether adding new capabilites gets cheaper, becomes iffy as we cant prove we have reusable pieces to reuse for a new capability
- What exactly is the difference between a weight-space atom and a function-space atom?
	- Weight space atom would modify the models weights not the input
	- Function space atoms directly operate on the internal representation to generate a delta to be added to the internal representation which means because that representation is learned too that the encoder and atoms can co-design the internal geometry
- What would count as an **E1 success**, separately from success of the entire Atoms project?
	- Well it would show that the accuracy pre decoding and post decoding is high on unseen combinations of the primitves it was not trained on.

--------------------------------------------------------------------------------------------
### Corrections
- Matrix and vector is not swapable terms:
	- A state is a vector, a point.
	- A matrix is a transformation that eats a vector and emits a vector
- Watch 3Blue1Brown's _Essence of Linear Algebra_

-----------------------------------------------------------
### Exam 1

- Your closed-map definition says an atom "outputs an answer near the manifold without help from the decoder." Stress-test it: healthy encoder, every input gets its own distinct code — and one atom that maps _every_ state to the encoding of [0,0,0,0,0,0,0,0]. Its output is always a perfectly valid code, no decoder involved. Under your definition, is that atom a closed map? Should it be? Repair the definition so it gives the right answer, and notice what your repair now has to mention that your original never did.
	- No its not a clsoed map and it shouldnt be but my definition would define it as one.
	- New definition: Atom outputs an answer in the manifold near the encoded ground truth result without the help of a decoder.
-   You wrote: "flexibility also creates more ways to fake factorization." Best sentence in the document — if it's yours. Prove it: describe one _concrete mechanism_ by which a composer free to route however it likes could score well on the task while no atom individually implements a primitive.
	- The composer could learn a matrix that works for the shape of any atoms output to result in the correct answer while the atom itself is not representing any primitive but rather the composer learned a matrix that gets correct answer with all the different atoms.
-   Your E1 success criterion is "high pre- and post-decode accuracy on unseen combinations." A0 achieves exactly that, at 0.95. Is A0's result an E1 success? If not, your criterion is missing a clause — state it.
	- It is missing a clause, without intermediate help from a teacher.
-   Your delta on the arms lists A1, A2, A3, and A0. One arm is missing — the one whose entire job is to catch the experiment lying to itself. Which, and what does it do?
	- A4, its job is to do negative testing to verify that the results are not just chance. So it destroys the question-answer relationship so the answer target is random to check if it still has high accuracy when its target is jsut noise

**Correction to Question 2**

-   You wrote: "flexibility also creates more ways to fake factorization." Best sentence in the document — if it's yours. Prove it: describe one _concrete mechanism_ by which a composer free to route however it likes could score well on the task while no atom individually implements a primitive.
	- The composer could learn a matrix that works for the shape of any atoms output to result in the correct answer while the atom itself is not representing any primitive but rather the composer learned a matrix that gets correct answer with all the different atoms.
	- **Delta:**
		- The real reason is that a composer can make it that an Atom performs only when preceeded by known partner.
	
-------------------- 