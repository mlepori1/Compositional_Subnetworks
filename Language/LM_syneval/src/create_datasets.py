import pickle as pkl
import numpy as np
import os

TASKS={
    ### Subject Verb Agreement
    "sn_task_1":  [
        "sn_task_1_sv_singular",
        "sn_task_1_s_singular",
        "sn_task_1_v_singular",
        "sn_task_1_sv_singular_ablate_v",  # In this case, only the subject is plural in the OOO sentences
        "sn_task_1_sv_singular_ablate_s"],  # In this case, only the verb is plural in the OOO sentences
    "sn_task_2": [
        "sn_task_2_sv_plural",
        "sn_task_2_s_plural",
        "sn_task_2_v_plural",
        "sn_task_2_sv_plural_ablate_v",  # In this case, only the subject is singular in the OOO sentences
        "sn_task_2_sv_plural_ablate_s"],  # In this case, only the verb is singular in the OOO sentences
    ### Reflexive Anaphora
    "sn_task_3": [
        "sn_task_3_pro_ant_singular",
        "sn_task_3_ant_singular",
        "sn_task_3_pro_singular",
        "sn_task_3_pro_ant_singular_ablate_pro",  # In this case, only the antecedent is plural in the OOO sentences
        "sn_task_3_pro_ant_singular_ablate_ant"],  # In this case, only the pronoun is plural in the OOO sentences
    "sn_task_4": [
        "sn_task_4_pro_ant_plural",
        "sn_task_4_ant_plural",
        "sn_task_4_pro_plural",
        "sn_task_4_pro_ant_plural_ablate_pro",  # In this case, only the antecedent is singular in the OOO sentences
        "sn_task_4_pro_ant_plural_ablate_ant"]  # In this case, only the pronoun is singular in the OOO sentences
}

def create_train_val_test(positive, negative, train_size=10000, val_size=500, test_size=1000):
    # Helper function to split data into train val and test splits
    # Each problem has 3 examples that follow a rule, one that doesn't
    np.random.shuffle(positive)
    np.random.shuffle(negative)

    train_positive = positive[:int(train_size * 3)]
    val_positive = positive[int(train_size * 3):int(train_size * 3) + int(val_size * 3)]
    test_positive = positive[int(train_size * 3) + int(val_size * 3):]
    train_neg = negative[:train_size]
    val_neg = negative[train_size:train_size + val_size]
    test_neg = negative[train_size + val_size:]

    train = []
    for i in range(len(train_neg)):
        pos_idx = int(3 * i)
        train.append([
            train_positive[pos_idx],
            train_positive[pos_idx + 1],                    
            train_positive[pos_idx + 2],
            train_neg[i]
            ])

    val = []
    for i in range(len(val_neg)):
        pos_idx = int(3 * i)
        val.append([
            val_positive[pos_idx],
            val_positive[pos_idx + 1],                    
            val_positive[pos_idx + 2],
            val_neg[i]
            ])

    test = []
    for i in range(len(test_neg)):
        pos_idx = int(3 * i)
        test.append([
            test_positive[pos_idx],
            test_positive[pos_idx + 1],                    
            test_positive[pos_idx + 2],
            test_neg[i]
            ])
    
    return train, val, test


def create_datasets(ab_sentences, a_flip_b_sentences, a_b_flip_sentences, train_size, val_size, test_size, part_val_size, part_test_size, dataset_names, base_dir):
    # Given two properties (a and b) generate datasets for compositional training, mask training on both, and testing on data iid with a partition of the compositional training
    base_negative_sents = int(train_size + val_size + test_size)
    base_positive_sents = base_negative_sents * 3

    part_negative_sents = int(part_val_size + part_test_size)
    part_positive_sents = part_negative_sents * 3

    # Create Base Datasets for Weight Training
    random_ab_sentences = np.random.choice(ab_sentences, base_positive_sents + (part_positive_sents * 2), replace=False) # positive sentences for base weight training and both partitions
    
    random_a_flip_sentences = np.random.choice(a_flip_b_sentences, int(base_negative_sents/2) + part_negative_sents, replace=False)
    random_b_flip_sentences = np.random.choice(a_b_flip_sentences, int(base_negative_sents/2) + part_negative_sents, replace=False)

    ab_base_set_positive = random_ab_sentences[:base_positive_sents]
    ab_base_set_negative = np.concatenate([random_a_flip_sentences[:int(base_negative_sents/2)], random_b_flip_sentences[:int(base_negative_sents/2)]]) # Get half the negative sentences from one set, half from the other

    ab_base_train, ab_base_val, ab_base_test = create_train_val_test(ab_base_set_positive, ab_base_set_negative, train_size=train_size, val_size=val_size, test_size=test_size)

    os.makedirs(os.path.join(base_dir, dataset_names[0]), exist_ok=True)

    train_path = os.path.join(base_dir, dataset_names[0], "train.pkl")
    pkl.dump(ab_base_train, open(train_path, "wb"))

    val_path = os.path.join(base_dir, dataset_names[0], "val.pkl")
    pkl.dump(ab_base_val, open(val_path, "wb"))

    test_path = os.path.join(base_dir, dataset_names[0], "test.pkl")
    pkl.dump(ab_base_test, open(test_path, "wb"))

    # Create Mask dataset for learning property a
    a_mask_set_positive = np.concatenate([
        np.random.choice(ab_sentences, int(base_positive_sents/2), replace=False),
        np.random.choice(a_b_flip_sentences, int(base_positive_sents/2), replace=False)
    ])

    a_mask_set_negative = np.random.choice(a_flip_b_sentences, base_negative_sents, replace=False)
    a_mask_train, a_mask_val, a_mask_test = create_train_val_test(a_mask_set_positive, a_mask_set_negative, train_size=train_size, val_size=val_size, test_size=test_size)

    os.makedirs(os.path.join(base_dir, dataset_names[1]), exist_ok=True)

    train_path = os.path.join(base_dir, dataset_names[1], "train.pkl")
    pkl.dump(a_mask_train, open(train_path, "wb"))

    val_path = os.path.join(base_dir, dataset_names[1], "val.pkl")
    pkl.dump(a_mask_val, open(val_path, "wb"))

    test_path = os.path.join(base_dir, dataset_names[1], "test.pkl")
    pkl.dump(a_mask_test, open(test_path, "wb"))

    # Create Mask dataset for learning property b
    b_mask_set_positive = np.concatenate([
        np.random.choice(ab_sentences, int(base_positive_sents/2), replace=False),
        np.random.choice(a_flip_b_sentences, int(base_positive_sents/2), replace=False)
    ])

    b_mask_set_negative = np.random.choice(a_b_flip_sentences, base_negative_sents, replace=False)
    b_mask_train, b_mask_val, b_mask_test = create_train_val_test(b_mask_set_positive, b_mask_set_negative, train_size=train_size, val_size=val_size, test_size=test_size)

    os.makedirs(os.path.join(base_dir, dataset_names[2]), exist_ok=True)

    train_path = os.path.join(base_dir, dataset_names[2], "train.pkl")
    pkl.dump(b_mask_train, open(train_path, "wb"))

    val_path = os.path.join(base_dir, dataset_names[2], "val.pkl")
    pkl.dump(b_mask_val, open(val_path, "wb"))

    test_path = os.path.join(base_dir, dataset_names[2], "test.pkl")
    pkl.dump(b_mask_test, open(test_path, "wb"))
    
    # Create ablation dataset distinguished by property a (partition of base dataset)
    a_partition_set_positive = random_ab_sentences[base_positive_sents : base_positive_sents + part_positive_sents]
    a_partition_set_negative = random_a_flip_sentences[int(base_negative_sents/2) :] # Use the remaining sentences in this split for creating partition sets

    _, a_partition_val, a_partition_test = create_train_val_test(a_partition_set_positive, a_partition_set_negative, train_size=0, val_size=part_val_size, test_size=part_test_size)

    os.makedirs(os.path.join(base_dir, dataset_names[3]), exist_ok=True)

    val_path = os.path.join(base_dir, dataset_names[3], "val.pkl")
    pkl.dump(a_partition_val, open(val_path, "wb"))

    test_path = os.path.join(base_dir, dataset_names[3], "test.pkl")
    pkl.dump(a_partition_test, open(test_path, "wb"))

    # Create ablation dataset distinguished by property b (partition of base dataset)
    b_partition_set_positive = random_ab_sentences[base_positive_sents + part_positive_sents : ]
    b_partition_set_negative = random_b_flip_sentences[int(base_negative_sents/2) :] # Use the remaining sentences in this split for creating partition sets

    _, b_partition_val, b_partition_test = create_train_val_test(b_partition_set_positive, b_partition_set_negative, train_size=0, val_size=part_val_size, test_size=part_test_size)

    os.makedirs(os.path.join(base_dir, dataset_names[4]), exist_ok=True)

    val_path = os.path.join(base_dir, dataset_names[4], "val.pkl")
    pkl.dump(b_partition_val, open(val_path, "wb"))

    test_path = os.path.join(base_dir, dataset_names[4], "test.pkl")
    pkl.dump(b_partition_test, open(test_path, "wb"))

if __name__=="__main__":

    np.random.seed(0)

    for task, subsets in TASKS.items():
        if "1" in task or "2" in task:
            # Subject Verb agreement data
            rawdata = ["obj_rel_across_anim.pickle",
            "obj_rel_across_inanim.pickle",
            "obj_rel_no_comp_across_anim.pickle",
            "obj_rel_no_comp_across_inanim.pickle",
            # Got rid of embedded SV agreement, so that the task is simply: Main Subject and Verb, rather than any noun and verb
            #"obj_rel_no_comp_within_anim.pickle",
            #"obj_rel_no_comp_within_inanim.pickle",
            #"obj_rel_within_anim.pickle",
            #"obj_rel_within_inanim.pickle",
            "prep_anim.pickle",
            "prep_inanim.pickle",
            #"sent_comp.pickle",
            "simple_agrmt.pickle",
            "subj_rel.pickle"]

            print(task)
            base_prefix = "singular_" if "1" in task else "plural_"
            flip_prefix = "plural_" if "1" in task else "singular_"

            sv_base_sentences = []
            s_base_v_flip_sentences = []
            s_flip_v_base_sentences = []

            for datafile in rawdata:
                datadict = pkl.load(open("../RawData/" + base_prefix + datafile, "rb"))
                for k in datadict.keys():
                    for (gram, ungram) in datadict[k]:
                        sv_base_sentences.append(gram)
                        s_base_v_flip_sentences.append(ungram)

                datadict = pkl.load(open("../RawData/" + flip_prefix + datafile, "rb"))
                for k in datadict.keys():
                    for (gram, ungram) in datadict[k]:
                        s_flip_v_base_sentences.append(ungram)

            sv_base_sentences = np.array(sv_base_sentences)
            s_base_v_flip_sentences = np.array(s_base_v_flip_sentences)
            s_flip_v_base_sentences = np.array(s_flip_v_base_sentences)

            # Create the base dataset for the task: training will be comprised on 9500 examples, not 10k, due to data limitations
            base_train_size = 9500
            base_val_size = 500
            base_test_size = 1000

            partition_val_size = 300
            partition_test_size = 300

            create_datasets(sv_base_sentences, s_flip_v_base_sentences, s_base_v_flip_sentences, base_train_size, base_val_size, base_test_size, partition_val_size, partition_test_size, subsets, "../datasets")

        else:
            # Reflexive Anaphora
            rawdata = [
            # Get rid of reflexives in sent complement. This reduces the problem to subject noun antecedents and pronouns
            #"reflexive_sent_comp.pickle",
            "reflexives_across.pickle",
            "simple_reflexives.pickle"]

            base_prefix = "singular_" if "3" in task else "plural_"
            flip_prefix = "plural_" if "3" in task else "singular_"

            pa_base_sentences = []
            a_base_p_flip_sentences = []
            a_flip_p_base_sentences = []

            for datafile in rawdata:
                datadict = pkl.load(open("../RawData/" + base_prefix + datafile, "rb"))
                for k in datadict.keys():
                    for (gram, ungram) in datadict[k]:
                        pa_base_sentences.append(gram)
                        a_base_p_flip_sentences.append(ungram)

                datadict = pkl.load(open("../RawData/" + flip_prefix + datafile, "rb"))
                for k in datadict.keys():
                    for (gram, ungram) in datadict[k]:
                        a_flip_p_base_sentences.append(ungram)

            pa_base_sentences = np.array(pa_base_sentences)
            a_base_p_flip_sentences = np.array(a_base_p_flip_sentences)
            a_flip_p_base_sentences = np.array(a_flip_p_base_sentences)

            # Due to small amonut of data, going to limit the train set to only 2500 examples, val to 200, and test to 200
     
            base_train_size = 2500
            base_val_size = 200
            base_test_size = 200

            partition_val_size = 200
            partition_test_size = 200

            create_datasets(pa_base_sentences, a_flip_p_base_sentences, a_base_p_flip_sentences, base_train_size, base_val_size, base_test_size, partition_val_size, partition_test_size, subsets, "../datasets")

