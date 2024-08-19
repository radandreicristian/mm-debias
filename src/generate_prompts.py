import itertools
import pandas as pd

# Define lists for the possible values
shot_types = ["close-up", "medium shot", "long shot"]
age_groups = ["young", "adult", "elderly"]
genders = ["male", "female"]
professions = ["barber", "coach", "businessperson", "football player", "construction worker", "manager", "CEO",
               "accountant", "commander", "firefighter", "mover", "software developer", "guard", "baker", "doctor",
               "athlete", "artist", "dancer", "mathematician", "janitor", "carpenter", "mechanic", "actor", "handyman",
               "musician", "detective", "politician", "entrepreneur", "model", "opera", "singer", "chief", "lawyer",
               "farmer", "writer", "librarian", "army", "real-estate developer", "broker", "scientist", "butcher",
               "electrician", "prosecutor", "banker", "cook", "hairdresser", "prisoner", "plumber", "attorney",
               "boxer", "chess", "player", "priest", "swimmer", "tennis player", "supervisor", "attendant",
               "housekeeper", "maid", "producer", "researcher", "midwife", "judge", "umpire", "bartender", "economist",
               "physicist", "psychologist", "theologian", "salesperson", "physician", "sheriff", "cashier", "assistant",
               "receptionist", "editor", "engineer", "comedian", "painter", "civil servant", "diplomat", "guitarist",
               "linguist", "poet", "laborer", "teacher", "delivery man", "realtor", "pilot", "professor", "chemist",
               "historian", "pensioner", "performing artist", "singer", "secretary", "auditor", "counselor",
               "designer", "soldier", "journalist", "dentist", "analyst", "nurse", "tailor", "waiter", "author",
               "architect", "academic", "director", "illustrator", "clerk", "police", "chef", "photographer",
               "drawer", "cleaner", "pharmacist", "pianist", "composer", "handball", "player", "sociologist"]
clothings = ["wearing work clothes", "wearing ethnic clothes"]
locations = ["at work", "at home"]

# Generate all combinations
combinations = list(itertools.product(shot_types, age_groups, genders, professions, clothings, locations))

# Format the combinations into the desired prompt format
prompts = ["{} photo of a {} {} {} {} {}".format(*combination) for combination in combinations]

print(f"Generated {len(prompts)} number of prompts")

df = pd.DataFrame({"Gendered Prompt": prompts, "Non-Gendered Prompt": prompts})

df["Non-Gendered Prompt"] = df["Non-Gendered Prompt"].str.replace("male ", "")
df["Non-Gendered Prompt"] = df["Non-Gendered Prompt"].str.replace("female ", "")

df.to_csv("prompts.csv", index=False)
