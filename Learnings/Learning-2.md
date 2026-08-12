
# LEARNING 2 — repayment log

## Unit 1 -- The Task With No ML
### Data.py
**Prediction - Why Data Python File Should do?**
- I think this python file should generate inputs  randomly based on a seed then run training and eval in a split that allows for sufficent combonations in training of primitves but with unseen ones left out purposefully for evaluation. If we have 8 primitves then the question becomes what are all the combinations whihc leads to another question needing answered: do we want at most 2 primitves as an input or more? I think we would want to cretae a data set of each primitive being trainined on its own a set number of times, then introduce 2 primitve tasks with some singletons still present then finally introduce three primitve tasks with small percantage of singletons and 2 primitive tasks then lastly train such that its an equal mix of the three types of tasks.  Each training example needs an 8 integer input, and a singleton or composition of primitves. A lingering question tho is bounding the compostions to at most 3 primitve tasks in a structure way the best option? Or is it more ideal to train on purely singletons or is it best to have a randomized amount of primitve tasks so one training round on seed 1 would generate 3 singletons, 2 two primitvives tasks, 3 more singletons, 5 eight primtive tasks, etc... with randomized mixing. In that case though we would want play around with setting the chances of certian types as more common like singletons or the like potentially.
- **Deltas**
	- E1 does not progress through singleton -> pair -> triple -> etc.. It only has a singleton or two-primitive task
	- E1 also does a trick where the model is always executing two-atom steps: if asking reverse then it does (reverse, identity)
	- Task object distinguishes between:
		- primitives: the true ground truth operation, length 1 or 2
		- instruction: always length 2 for the model (if singleton then identity is added)
			-  We did this to avoid asking "can the system learn termination?" but since our composer is intruction-conditioned so the instruction is a sequence of primitves one per step so variable depth doesnt require learned halting at all.
			- Now still a cost to having variable depth and that is defining unseen while managing the data so we dont end up with the problem of two recipes that are differen result in the same dish. Deining the depth and limiting to 2 made this computable. Easy fix tho is to change our primitives liek we started to in v2 to make primitves not contain self cancel or absorb properties
	- We dont add variable lengths because it asks another question: Can the system learn variable-depth compostion and termination?
		- The dataset is organized into 4 useful buckets:
			- Train
				- 8 singleton tasks + 40 training primitive pairs
			- Seen_Heldout
				- same 40 training pairs but completly fresh input sequences
			- Unseen
				- 24 primitive pairs never used for training
			- Singleton Eval
				- fresh examples of each individual primitive
---------------------------------------------
**Q1 - Why do we need `seen_heldout` examples of already-trained compositions if we also have completely unseen compositions?**
We need seen_heldout because we want to limit the possiblity that an atom learns how to operate on only seen/expected inputs, just like it can learn to only perform on expected primitve combos it can equally learn to work on only specific inputs.

**Seen_heldout diagnoses the whole learning system not just Atoms**
Two seperate failure modes:
- seen_heldout fails
	- model may have memorized particular input/output examples
- seen_heldout succeeds but unseen fails
	- it can generalize to new inputs but not to new primitive combos
- both succeed
	- much stronger evidence for real compositional generalization
--------------------------------------------------
**Split Construction**
First we I need to understand leakage, with our 8 primitives we can get a result where different recipes result in same meal.
- Example: 
	- input = [1,5, 8, 3] and asked to run [reverse, increment] = [4, 9, 6, 2]
	- input = [1,5, 8, 3] and asked to run [increment, reverse] = [4, 9, 6, 2]

This is Extensional Equivalence, when two tasks may be described differently but they produce the same output for every relevant input. The code fixes this by approximation, we generate a large fixed probe appyling every singleton and every ordered pair to those same inputs, and using the resulting outputs as a signature. Pair sharing the same signature are grouped into one class then the entire class goes into train or held-out together. 

But there is a sneaky second leakage problem, suppose a pair computes exactly the same function as a singleton? 
- Example:
	- input: [1, 4, 2, ...] and asks [rotate_left, sort_asc]
	- Because sorting destroys the ordering of information created by the rotate then if sorting is already trained as a singleton then putting rotate first into a held out would again fake novelty as the model has already trained on that result.

So any Pair-class equivalent to a singlton is forced into training and never held out. These two fixes only works efficiently because we limit the size of primitves in compositions to 2.

Since we have size 2 of compositional primitves then that means the split requires every primitive to appear at least twice, once in position 1 and once in positon 2. But there is a sneaky problem in that, if the second primitve doesnt do anything meaningful with the output of the first primitve then that pair didnt really teach the system anything about doing the first primitive with another meaningful operation.
- Example:
	- reverse -> identity, identity doesnt rely on reverse not really
	- rotate_left -> sort_asc, sorting is correct regardless of rotating that happens before it

Data file fixes this by defining informative pairs and imposes the coverage requirement again using only those pairs. Rule: for each non-identiy primitive there must be at least 2 informative appearences in each position. 

So the Split protects against 3 things:
- Literal overlap in training and unseen
- Functional overlap where the result of a pair is the same between two different pairs
- Fake coverage where primitive is there but its contributuion is meaningless

-------------------------------------------------------------
**Q2 - Reason Thru these**
- Why is it important that every primitive appears in **both position 1 and position 2** during training?
	- This is important so that an Atom isnt inclined to learn to work just as the first atom used or as just the second. Basically so they dont become dependent on position in the query.
- Why isn't merely counting appearances enough? Explain what an **informative occurrence** buys us.
	- Counting appearences isnt enough because some of our primitives dont mutate the numbers themselves and some primitives go even further with not only not mutating the integers but also destroys the ordering of the information. This results in a situation where the second primitive doesnt really need the first primitive or its contribution isnt very relevant. So informative occurence is the idea that the code informs itself which primitives act this way and then excludes them from occurence count which results in avoiding this problem.
- Suppose we have: Train = reverse -> identity, Held Out = reverse -> increment....Why would saying “reverse appeared during pair training, so reverse had sufficient compositional experience” be misleading?
	- Because reverse in training never paired with a meaningful partner and never relied on a meaningful partner, it always acted as a singleton in training.

**Deltas**
- In the question: "Why isn't merely counting appearances enough? Explain what an **informative occurrence** buys us."
	- My answer said the code informs itself which primitives act this way, where the code doesnt actually do that for whole primitives but rather for pairs. 
		- Example where that would be wrong:
			- rotate_left itself is not uninformative primitive but when in a pair like so [rotate_left -> sort_asc] it is but only because the final result is the same as if you just ran sort_asc. Do this, [rotate_left, reverse] and it is no longer uninformative, it actually results in change of the final outcome.

Raw occurence asks whether a primitives name appears, informative asks whether its participation matters.

-----------------------------------------------------
**build_split() Machinery**
Conceptually:
- Compute all functional equivalence classes, so calculate the pairs and use there results as a signature to group together in a functional equivalence class. (Do we do this for every input we plan on running as well with every pair?)
	- We do not calcualte on every input, we actually generate a probe set of 20k 8 digit inputs that runs on every combo of pairs
- Identitfy classes that must stay in training (What determines that?)
	- This is determined by calculating the signatures of singletons and then if a class signature matches a singletons it is forced as training because all singletons are training tasks so if they were in held out then it wouldnt actually be unseen
- Randomly shuffle remaining classes, so to make sure we get a nice randomized data set
- Select whole classes until exactly 24 pairs are held out, the held out is the unseen for eval
- Test all coverage constraints
- If any constraint fails, throw that candidate away
- Try again, this happens because pairs dont evenly seperate into classes so one class might have 3 pairs another might have 2 so we might end up with 22 held out pairs on accident so we have to try again or simply a constraint might fail like "increment only appears once in position 2" or informative coverage fails.

Important part is randomness proposes a split then constraints clarify and verify it.

------------------------------------------------------------------------------------
**Q3 - Forget all these constraints. Randomly make 100 different 40/24 splits, run the model on all of them, and report the average. Would that solve leakage and coverage problems?**
Coverage yes, leakage no....I think the important part there is while it would smooth out the coverage which I think would solve that problem. Leakage is a bit more than seeing different combinations, its having the eval set be unreliable as a measurement.

**Deltas**
- Coverage wouldnt really be covered either not in the way we need:
	- Becuase averagign over 100 random splits can smooth the average frequency but does not gurantees that any given split is valid. This would result in less clean data by mixing in clean runs with dirty ones which would be harder to pull apart and see the limiting or driving factors.

--------------------------------------------------------------------------
### Make_split.py

**Prediction - What do I expect this file to do?**
Well we know it doesn't contian the split logic itself as we have build_split() function in data.py that does that, so this file is is most likely in charge of calling the functions in data.py and get the split for any given seed to then save to disk. I dont think any verification should be neccassary at this level as it happens inside the data file, so this should primarily serve as a binding for getting the same split for the same seed.

**Deltas**
- Actual code does re validate the split in this file even thought its validated in the level before it.
	- So in data it verifys i constructed something that satisfys the rules, and here it validates the finished object to make sure required properties actually hold.
- Checks if split file exisits alreayd and wont re generate it as the files hash is recorded to guratnee consitency when using the same seed.

This script is doing 3 jobs:
- generate, make a candidate split
- certify, verify its contract
- freeze, write an identifyable reproducible artifact

---------------------------------------------
### Read the actual split artifacts

**Prediction - Write down the fields you think must be stored so someone months later could audit exactly what split was used without rebuilding it**.

Fields:
- Seed that generated this Split
- Primitive set used
- Experiment/s it was used for
- Seen vs unseen split itself
- Seen vs unseen ratio
- Pairs classified as train only

**Actual split stores for v1**
- split_seed
- train_pairs, exact 40 pairs
- heldout_pairs, exact 24 pairs
- n_train_pairs, 40 (how many)
- n_heldout_pairs, 24 (how many)
- position1_counts_train, for each primitive how many training pairs use that as first operation
- position2_counts_train, for each primitive how many training pairs use that as second operation
- informative_position1_counts_train, for each primitive how many meaningful composition pairs use it first excluding pairs whose overall result is equal to a singleton
- informative_position2_counts_train, for each primitive how many meaningful composition pairs use it second excluding pairs whose overall result is equal to a singleton
- heldout_class_ids, The IDs of the functional-equivalence classes that were assigned entirely to the held-out set
- n_distinct_pair_functions, the number of uniqie results produced by all 64 ordered primitive pairs
- n_distinct_functions_heldout, the number of unique results represented by the 24 eval pairs
- n_classes_forced_train, the number of pair-equivalence classes that had to stay in training because their function was identical to a singleton task already used in training.
- signature_seed, the random seed used to generate the fixed prob inputs used for comparing two tasks compute the same result
- signature_samples, number of probe inputs used 
- attempts, how many times it had to retry
- constraints, human readable record of rules a candidate split had to satisfy
- plus task world metadata like n_primitives = 8, seq_len = 8, vocab = 10

------------------------------------------------------------------------
**Q4 - Why can there be 24 held-out pairs but only 17 distinct held-out functions without compromising the unseen-composition test?**
This can be true because the main problem trying to be solved by seeing how many unique functions/results is to make sure we dont get one with the same in both training and eval. So if they are just in eval then its a non issue in theory but we might also later want to have cleaner held_out data later.

____________________________________________________________________________________
### DECISIONS - D2

**Prediction - What do I expect D2 to contain in the DECISIONS.md file? Focus on: what problem a naive 40/24 split created, why functional equivalence mattered, what rule was adopted to fix it, and what tradeoff or limitation might remain even after the fix.**
Considering this is the first decision related to split design we are looking at, I predict that it was to decided to add in the functional-equivalence classes in order to fix the naive 40/24 split that would lead to multiple possible leakage and coverage issues. After this fix tho what would remain is a sneakier leakage where when two primitives are used together and one does not contribute meaningfully to the output or where the function of the pairs is equal to the function of a singleton resulting in heldout not always being a true unseen eval group.

**Deltas**
- D2 also catches the case where when two primitives are used together and one does not contribute meaningfully to the output or where the function of the pairs is equal to the function of a singleton resulting in heldout not always being a true unseen eval group.
- What does remain after D2 is a coverage problem where raw positional counts not representing informative compositional exposure. 

------------------------------------------------------
### Decisions - D3
**Thought Experiment**
- T4 wants to check that no primitive can be recreated by composing two _other_ primitives.
But our primitive set contains:

	-   `identity`
	-   `reverse`
	-   `reflect`
	-   `swap_halves`

Now answer this without looking anything up:

**Does `identity` satisfy T4 as written? Why or why not?**
No because if we give it reverse -> identity that is the same as reverse, and that is true for everything that uses identity. Identity does not rely contribute anything meaningful to the pairs function.

**Delta**
- Answered different identity problem, rather the issue is reverse -> reverse = identity. T4 asks "can any primitive be reproduced by coposing two other primitives?" And with identity the answer is yes. 
- D3 makes the decision to exclude identity as a T4 target while still allowing it to appear as a component of compositions.

-----------------------------------------------------------------------
### Decisions - D10/D13/D14/D15
- D10: sort_asc is structurally troublesome
	- Doesnt replace but identitfys it as a weak link due to it being both idempotent and order destroying and is main culprit for only having 39 distinct functions among 64 ordered pairs
- D13: the split technically satisfied coverage but some coverage was fake
	- raw coverage not equal to useful coverage gap identifyied specifically with identity
- D14: redefine coverage using informative pairs
	- A pair is uninformative when its overall function equals the same as a singleton so we add rule every non-identity primitive must appear at least twice ine ach position among informative training pairs but keep identity exempt
- D15: measure whether sort_asc actually damages the evaluation enough to justify replacing it
	- Decide after measuring that the improvement was minimal so keep it for now

**Q5 - Suppose a new split has zero functional leakage and every primitive appears four times in each position, but all four appearances of `increment` are singleton-equivalent pairs. Is that split valid under D2? Under D14? And why?**
Under D2 that split would still be valid, under D14 it would not because D14 introduces the rule of informative pairs, which requires that the function of a pair be unique to that of a singleton for held_out eval set.

**Delta**
- D14 doesnt actually add a new rule about held-out set, it just says there must be at least 2 appearences in each position in training pairs where the function is not equal to the singletons for every non-identity primitive.

-----------------------------------------------------------------------
### Walkthrough of Unit 1 Pipeline from primitive definitions to evaluation
First we define 8 primitives: [`identity`, `reverse`, `increment`, `sort_asc`, `rotate_left`, `swap_halves`, `double`, `reflect`], we define the functions for each which are then used as ground truth during training and evaluation. We run 1 primitive at a time and have method that does that so we can easily have another method that can take in a composition like, [reverse, increment], and run the primitive function for each using the previous result as the starting point for next. Now for generating training and eval data we first need to define and generate a split to use some for training and some for evaluation. We needed to consider that the eval set needs to have compostions that are not present in the training but we cant do this naivly by just splitting the random compositions generated by a seed. We need to create functional equivalence classes to group together pairs that result in same function by generating a large random amount of inputs and calculating the function of each pair composition to use as a signature. This tho still leaves some gaps, for instance since singletons are always in training set then any functional equivalence class whos function matches a singleton has to be put into training set. This solves our leakage problem but still leaves us with a coverage problem, if we dont define that each primitive needs to show up in postion 1 and 2 of a pair x amount of times we might end up with a system that learns primitives but only in specific positions so we must add a gate that requires each primitive to be in position 1 and 2 twice within the training set. But we do need to add one more rule, identity as a primitive can have the same fucntion as two other primritives, so we need to exclude it from the gate checking that no primitive can already be reproduced by composing two other primtives because for singletons in training we pad it with identity. Once we generate the dataset tho we want to verify it meets constraints and if it doesnt retry, and then we check again that it is valid and if it is we save it indexing its hash and saves the training/held_out sets along with seed used for gneerating them, seed used for generating the probes for input, the constraints used, number of inputs smapled, number of pairs in each set, etc... so that it is all auditable. Now we also generate seen_heldout which is there to verify the same compositons that it trained on but with new inputs so to diagnose if it is memorizing inputs rather than primitives. So if seen_heldout is trong but unseen s weak then we can confidently say it generalizes on inputs but is memorizing on compositions.

**Q6 - Why does D14 require informative coverage in both position 1 _and_ position 2, rather than merely requiring each primitive to participate in four informative training pairs anywhere?**
D14 requires informative coverage on both postions because the atoms may learn to only operate as either the starting transformation or the ending one and not learn true reusability.

-------------------------------------------------------------
