We will have three long horizon tasks, These tasks will include multiple tasks which are predefined in the RLBench, 
We need


### Notes:
1. Check whether the post condition is satisfied, before checking preconditions.
2. At every step, we check post condition, if its true we skip the contract.
3. The HRL handles all skills.
4. The High Level Planner handles everything except Reach.
5. For Reached Precondition : We send it to the HRL
	1. The HRL May directly call the reach skill
	2. or it may need to do some pick and place before reaching
6. For The precondition of the Place skill the target must be clear
	1. if the target is not clear, and reachable, The HRL 
		1. Places current obj in temp location
		2. Picks and Places things from top of the target
		3. Clears the target
	2. If the target is not clear and not reachable
		1. The HRL tries stuff till budget is exhausted
		2. Reprompt the VLM for a new sequence.
	3. If the target is clear and not reachable
		1. The precondition is already satisfied
		2. The Contract tries to execute the place skill
		3. The place skill fails,
		4. Reprompt the VLM for a new sequence.
7.  For the precondition reached of Open it is same as precondition reached of Pick
8. For the precondition of Object Sweep, The HRL should remove the objects in the way, 
9. For clear preconditon of Open, we ask HRL to do pick and place.

### Total Skills
Open -> Grill / Box / Oven
Reach -> ALL Objects
Grasp  -> Meat / Vegetablels on Weighing Scale /  Items inside the Box
Place -> All the items in Grasp
Close -> Grill / Box / Oven

#### We can have Five skills offered to the High Level Planner 

Open -> Grill / Box 
Grasp  -> Meat  /  Items inside the Box / Groceries 
Place -> All the items in Grasp
Close -> Grill / Box 
Reach - > Everywhere

The tasks which we are using are
1. Open / Close Grill
2. Open  / Close Box
3. Put Meat on Grill
4. Put Meat off Grill
5. Put Shoe inside box
6. Put items inside cupboard  -> Restricted Version with only three groceries
7. Take Plate off coloured Dish rake 

Example Long Horizon Tasks

1. Heat the Meat and Plate it (Open Grill + Put Meat on Grill + Close Grill + Take Meat off Grill + Take Plate off coloured Dish rake + Put Meat on Plate)
	1. Open Grill  (Single Contract)
	2. Put Meat on Grill  ( Three Contracts (Reach Pick and Place))
	3. Close Grill  (Single Contract)
	4. Take Meat off Grill ( Three Contracts (Reach Pick and Place)
	5. Take Plate off coloured Dish rake (Three Contracts (Pick and Place))
	6. Put Meat on Plate (Three Contracts (Reach Pick and Place))
2. Tidy the table (
	1. Place All Groceries in Cupboard  (Three Groceries) ( 9 Contracts) (Reach Pick and Place)
	2. Open Box (Single Contract)
	3. Put Shoes inside Box ( Three contracts) (Reach Pick and Place)
	4. Close the Box (Single Contract)



## Pre Conditions:

### Reach Contract
1. Reachable | Object
### Pick Contract 
1. Gripper Empty
2. Reached | Object --> "ee_at(grasp_pose)"

### Place Contract
1.  "Reached (Target)"
2.  "holding(object)"
3.  "clear(target)" -> Nothing on Top of or inside the target

### Open Contract
1. Reached | Object ->  "ee_at(handle|object)",
2. "Holding Nothing" -> Gripper Open
3. Clear (Object)


### Close Contract
1. "ee_at(handle|object)"
2. Holding Nothing -> Gripper Open



## Post Condition

### Reach Contract
1. Reached Object
### Pick Contract
1. "holding(object)"

### Place Contract
1. "Object on Target"

### Open Contract
1. "object_state(object)=open"

### Close Contract
1. "object_state(object)=Close"



