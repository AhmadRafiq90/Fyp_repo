# Plant Species Classification - Step by Step Model Development

A simple project that teaches computers to recognize different plant species from leaf pictures. We started with basic models and kept improving them until we got really good results!

## 🎯 What This Project Does

This project shows how we built a smart system that can look at a leaf picture and tell you what type of plant it is. We tried many different approaches and kept making them better, going from 65% accuracy to 90% accuracy!

## 📊 Our Journey: From Worst to Best Models

### Step 1: Simple LeNet Model → 65% Correct Answers
**File:** `LENET.py`

**What we did:**
- Built a basic computer vision model with simple layers
- **Convolutional Layers:** These look at small parts of the image to find patterns (like edges and shapes). Think of it like having many tiny magnifying glasses scanning different parts of a leaf photo.
- **Max Pooling:** This makes the image smaller while keeping important information. Like taking a high-resolution photo and making a smaller version that still shows all the important details.
- **Dense Layers:** These make the final decision about what plant it is by combining all the features found earlier.

**Problem:** The model was too simple and couldn't learn complex leaf patterns well. It's like trying to recognize faces with your eyes closed - you just don't have enough information processing power.

### Step 2: LeNet + Data Tricks → 67% Correct Answers
**File:** `LENET.py` (improved version)

**What we added:**
- **Data Augmentation:** We created more training pictures by rotating, zooming, and flipping existing leaf photos
- Think of it like teaching a kid to recognize cats by showing them cats from different angles, distances, and lighting conditions
- This helps prevent **overfitting** - when a model only works on pictures that look exactly like the training photos

**Problem:** Still not good enough - the basic model design was holding us back. No amount of extra photos could help a calculator do the job of a computer.

### Step 3: AlexNet (Bigger Brain) → 80% Correct Answers
**File:** `Alex_Net.py`

**What we did:**
- Used a much bigger and smarter model called AlexNet
- **Bigger Filters:** Used large filters (11x11 pixels) to see bigger patterns in leaves. Small filters (3x3) can only see tiny details like edges, but big filters can see entire leaf shapes.
- **Deeper Network:** More layers means the computer can learn more complex features
  - Layer 1: Finds edges and lines
  - Layer 2: Combines edges to find shapes
  - Layer 3: Combines shapes to find leaf parts
  - Final layers: Combines everything to identify the plant

**Why it worked better:** Like upgrading from a basic calculator to a computer - more processing power means better results!

### Step 4: AlexNet + Better Training → 82% Correct Answers
**File:** `enchanced_alex_net.py`

**What we added:**
- **Batch Normalization:** This helps the model learn faster and more stable (like giving it better study habits)
  - Without it: Model learning is like a student having good days and bad days randomly
  - With it: Consistent learning progress every day
- **Better Regularization:** Added dropout to prevent the model from cheating (memorizing instead of actually learning)

**Dropout explained:** Randomly turns off some brain cells during training so the model can't rely on just a few neurons. It's like practicing for a team sport where random players might not show up - everyone has to learn to do multiple jobs.

### Step 5: AlexNet + Smart Loss Function → 85% Correct Answers
**File:** `alex_net_with_improved_regularization.py`

**What we added:**
- **Focal Loss:** Pays more attention to difficult examples that the model gets wrong
  - Regular training: All mistakes are treated equally
  - Focal loss: Hard mistakes get more attention, easy examples get less
- **Class Balancing:** Makes sure the model learns all plant types equally (not just the common ones)
  - Problem: If you have 1000 rose pictures but only 10 cactus pictures, model only learns roses well
  - Solution: Give equal importance to both during training

Think of it like a teacher spending extra time with struggling students while letting advanced students work independently.

### Step 6: EfficientNet (Too Smart for Its Own Good) → Failed!
**Files:** `EfficientNetB0(pretrained).py`, `customefficientNet.py`

**What we tried:**
- **Transfer Learning:** Used a model that was already trained on millions of images
  - Like hiring someone who already knows 1000 different jobs to learn one specific new job
  - Should be faster and better than training from scratch
- **EfficientNet:** A very smart model that balances width (how many neurons per layer), depth (how many layers), and resolution (image size) perfectly

**Why it failed:** Like using a rocket ship to go to the grocery store - too powerful for our simple task and kept overfitting. The model was so smart it memorized every training image instead of learning general patterns.

### Step 7: Super Advanced Attention Model → Gave Up!
**File:** `random/CBAM_with_special_attention_to_minority_classes.py`

**What we tried:**
- **CBAM Attention:** A system that tells the model exactly where to look in the image (like pointing at important parts)
  - Channel attention: "Pay attention to color information, ignore texture"
  - Spatial attention: "Look at the center of the leaf, ignore the background"
- **Extreme Data Balancing:** Made sure every plant type had exactly the same number of examples
  - Created synthetic examples of rare plants
  - Removed examples of common plants

**Why we quit:** Needed too much computer power - even our powerful RTX 2070 Super couldn't handle it! Training would take weeks instead of hours.

### Step 8: ResNet34 (The Sweet Spot) → 90% Correct Answers! ✅
**File:** `resnet34(leaf_classifier).py`

**What made it work:**
- **Residual Connections:** Like giving the model shortcuts to remember earlier information
  - Problem with deep networks: Information gets lost as it passes through many layers
  - Solution: Add shortcuts that carry original information directly to later layers
  - Like having a telephone game where you can also whisper the original message directly
- **34 Layers Deep:** Deep enough to learn complex patterns but not so deep it gets confused
- **Perfect Balance:** Not too simple, not too complex - just right!

**Why ResNet works better:**
- Can train very deep networks without losing information
- Learns both simple features (edges) and complex features (leaf shapes) effectively
- Generalizes well to new images it hasn't seen before

### Step 9: Two-Step Smart System → Final Solution! 🎯
**Files:** `resnet18(binary_classifier).py` + `resnet34(leaf_classifier).py`

**How it works:**
1. **Step 1 - Is this a leaf?** First model checks if the picture actually has a leaf in it
   - Input: Any image (could be a leaf, laptop, person, anything)
   - Output: "YES, this is a leaf" or "NO, this is not a leaf"
2. **Step 2 - What type of leaf?** If yes, second model identifies the plant species
   - Input: Image confirmed to contain a leaf
   - Output: Specific plant species (Rose, Oak, Maple, etc.)

**Why we needed this:** 
- Without this, the model would try to classify everything as a plant (even laptops!)
- Like having a bouncer at a plant party - only leaves get in!
- **Real-world problem:** People might upload random photos, and we don't want the system saying "Your laptop is a rare species of fern!"

**Binary Classification explained:** Instead of choosing between 40+ plant species, the first model only has to make a simple yes/no decision. This is much easier and more reliable.

## 🔄 What We Learned

- **Start Simple:** Begin with basic models before trying complex ones (like learning to walk before running)
- **Overfitting is Bad:** When models memorize instead of learning (like students who cheat on tests - they fail when given new problems)
- **Balance is Key:** Too simple = bad results, too complex = uses too much power and overfits
- **Real World is Messy:** Models need to work on random internet photos, not just perfect lab pictures
- **Computational Limits Matter:** Even the best ideas are useless if your computer can't run them
- **Multi-stage Systems Work:** Sometimes solving one big problem by breaking it into smaller problems works better

## 🚀 How to Use Our System

### To Classify a Plant:
```python
# First, check if it's actually a leaf
python resnet18(binary_classifier).py

# If it's a leaf, find out what plant it is
python resnet34(leaf_classifier).py
```

### To Test the Models:
```python
python Test_models/runModel.py
```

## 📁 What's in This Folder

- **Main Folder:** All our different model attempts
- **`Data_Preprocessing_and_cleaning/`:** Tools to clean and prepare plant pictures
- **`Test_models/`:** Scripts to test how good our models are
- **`random/`:** Failed experiments we tried (but kept for learning!)

## 🎯 Our Final System

**Two-Step Process:**
1. **Leaf Detector (ResNet18):** "Is this a leaf or not?"
2. **Species Classifier (ResNet34):** "What type of plant is this leaf from?"

**Final Score:** 90% accuracy - that means it gets 9 out of 10 leaf pictures correct!

**Why this works so well:**
- Specialization: Each model has one specific job it does really well
- Error Prevention: Stops false classifications of non-plant objects
- Robust: Works on messy real-world photos from the internet

## 📚 Simple Definitions

- **Model/Network:** The computer brain that learns to recognize plants
- **Training:** Teaching the computer using thousands of example pictures (like showing a child 1000 dog photos so they learn what dogs look like)
- **Accuracy:** How often the computer gets the right answer (90% = 9 out of 10 correct)
- **Overfitting:** When the computer cheats by memorizing instead of truly learning (like memorizing test answers without understanding the subject)
- **Layers:** Different parts of the computer brain that each do specific jobs (like an assembly line where each worker does one task)
- **Features:** Patterns the computer learns to recognize (like leaf shapes, colors, textures, vein patterns)
- **Generalization:** How well the model works on new, unseen images (the real test of learning)
- **Binary Classification:** A simple yes/no decision (much easier than choosing between many options)
- **GPU (Graphics Card):** Special computer chip that's really good at the math needed for AI (like having a calculator that can do 1000 calculations at once)

## 🤔 Why Some Things Failed

- **EfficientNet:** Too smart for our dataset size - like bringing a PhD professor to teach kindergarten math
- **CBAM:** Great idea but needed supercomputer-level resources we didn't have
- **Simple models:** Not smart enough to learn complex leaf patterns - like asking a calculator to write poetry

---

*This project shows how building AI is like learning to ride a bike - you start wobbly, fall down a few times, but eventually get really good at it! The key is learning from each failure and making improvements
