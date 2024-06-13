import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#from keras.models import Sequential
#from keras.layers import Dense

#pickle write
'''def test_pkl_serial(dir):
    for filename in os.listdir(dir):
        new_name = filename.replace('.png', '.pkl')
        new_name = os.path.join('pkl', new_name)
        full_path = os.path.join(dir, filename)
        img = cv2.imread(full_path)
        with open(new_name, 'wb') as f2:
            pickle.dump(img, f2)


start = time.time()
test_pkl_serial('dataset_test')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#pickle reads
'''def check_pkl_deser(dir):
    for filename in os.listdir(dir):
        full_path = os.path.join(dir, filename)
        with open(full_path, 'rb') as f:
            pickle.load(f)


start = time.time()
check_pkl_deser ('pkl/')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#numpy write
'''def test_npy_ser1(dir):
    for filename in os.listdir(dir):
        new_name = filename.replace('.png', '.npy')
        new_name = os.path.join('npy', new_name)
        full_path = os.path.join(dir, filename)
        img = cv2.imread(full_path)
        with open(new_name, 'wb') as f2:
            np.save(f2, img)


start = time.time()
test_npy_ser1('dataset_test')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#Numpy read
'''def check_npy_des1(dir):
    for filename in os.listdir(dir):
        full_path = os.path.join(dir, filename)
        with open(full_path, 'rb') as f:
            np.load(f)


start = time.time()
check_npy_des1('npy/')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#NumPy write 1 file
'''def test_npy_ser2(dir):
    np_arrays = []
    for filename in os.listdir(dir):
        full_path = os.path.join(dir, filename)
        np_arr = cv2.imread(full_path)
        np_arrays.append(np_arr)
    np.savez('images_dataset.npz', *np_arrays)


start = time.time()
test_npy_ser2('dataset_test')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#NumPy read 1 file
'''def check_npy_des2():
    data = np.load('images_dataset.npz', allow_pickle=True)


start = time.time()
check_npy_des2()
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#pydantic write
'''def test_pyd_ser(dir):
    np_arrays = []
    for filename in os.listdir(dir):
        full_path = os.path.join(dir, filename)
        np_arr = cv2.imread(full_path)
        np_arrays.append(np_arr)
    np.savez('images_dataset.npz', *np_arrays)


start = time.time()
test_pyd_ser('dataset_test')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#pydantic read
'''def check_pyd_des(dataset_path):
    file = open(dataset_path)
    dataset_dict = json.load(file)
    dataset = IAMDataset(**dataset_dict)
    images = []
    for word in dataset.words:
        image_bytes = base64.b64decode(word.image)
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        images.append(image)


start = time.time()
check_pyd_des('dataset.json')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''


#h5py write
'''def test_h5py_ser(dir):
    images = []
    for filename in os.listdir(dir):
        full_path = os.path.join(dir, filename)
        np_arr = cv2.imread(full_path)
        images.append(np_arr)

    resized_images = [cv2.resize(img, (64, 64)) for img in images]
    images_array = np.stack(resized_images, axis=0)
    with h5py.File('h5py_images.h5', 'w') as hdf5:
        hdf5.create_dataset('images', data=images_array)

start = time.time()
test_h5py_ser('dataset_test')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''

#DataSet preprocess
'''file_text_map = {}
def process_xml():
    for filename in os.listdir('C:/Users/anton/PycharmProjects/MyProject3/data/paths/'):
        root = ET.parse(os.path.join('C:/Users/anton/PycharmProjects/MyProject3/data/paths/', filename)).getroot()
        handwritten_part = root.find('handwritten-part')
        if handwritten_part is None:
            continue

        for line in handwritten_part.findall('line'):
            for word in line.findall('word'):
                id = word.attrib.get('id')
                text = word.attrib.get('text')
                parts = id.split('-')
                fullpath = os.path.join('C:/Users/anton/PycharmProjects/MyProject3/data/images/', parts[0], "-".join(parts[:2]), "-".join(parts) + '.png')
                if os.path.getsize(fullpath) > 0:
                    file_text_map[fullpath] = text


def process_images(image_width=128, image_height=64):
    with h5py.File('h5py_images.h5', 'w') as hdf5:
        for file in tqdm(file_text_map, desc="Parse Images"):
            text = file_text_map[file]
            image = preprocess(file, image_width, image_height)
            file_root, file_extension = os.path.splitext(file)
            group = hdf5.create_group(os.path.basename(file_root))
            group.create_dataset('image', data=image)
            group.create_dataset('text', data=text)


process_xml()
process_images()'''

#h5py read
'''def check_h5py_des():
    with h5py.File('h5py_images.h5', 'r') as hdf5:
        images_array = hdf5['images'][:]


start = time.time()
check_h5py_des()
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''



#PC read
'''def test_pc_ser(path):
    for file in os.listdir(path):
        fullpath = os.path.join(path, file)
        cv2.imread(fullpath)


start = time.time()
test_pc_ser('dataset_test')
end = time.time() - start
end_str = "{:.10f}".format(end)
print(end_str)'''


'''def choose_random_images_from_nested_folders(root_folder, target_folder, num_images=100):
    # Проверка и создание целевой папки, если она не существует
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    chosen_images = []
    image_extensions = ('.png', '.jpg')

    while len(chosen_images) < num_images:
        try:
            # Список всех подпапок в корневой папке
            first_level_folders = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if
                                   os.path.isdir(os.path.join(root_folder, d))]
            if not first_level_folders:
                raise ValueError("No subfolders found in the root folder")

            # Случайно выбрать первую папку
            first_folder = random.choice(first_level_folders)

            # Список всех подпапок во второй папке
            second_level_folders = [os.path.join(first_folder, d) for d in os.listdir(first_folder) if
                                    os.path.isdir(os.path.join(first_folder, d))]
            if not second_level_folders:
                raise ValueError("No subfolders found in the first chosen folder")

            # Случайно выбрать вторую папку
            second_folder = random.choice(second_level_folders)

            # Список всех изображений во второй папке
            images = [os.path.join(second_folder, f) for f in os.listdir(second_folder) if
                      f.lower().endswith(image_extensions)]
            if not images:
                continue

            # Случайно выбрать изображение
            chosen_image = random.choice(images)
            if chosen_image not in chosen_images:
                chosen_images.append(chosen_image)

        except ValueError as e:
            print(e)
            break

    # Переместить и переименовать изображения
    for i, image_path in enumerate(chosen_images):
        # Получить расширение файла
        _, ext = os.path.splitext(image_path)
        new_name = os.path.join(target_folder, f"{i + 1}{ext}")
        shutil.move(image_path, new_name)
        print(f"Перемещено и переименовано: {image_path} -> {new_name}")

    print(f"Успешно перемещено {len(chosen_images)} изображений.")


# Пример использования
root_folder_path = './dataset_sent'
target_folder_path = './save_sent'
choose_random_images_from_nested_folders(root_folder_path, target_folder_path, 100)'''





'''def calculate_wer_cer(file1_path, file2_path):
    # Открытие файлов и чтение строк
    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    # Проверка, что количество строк в обоих файлах одинаковое
    if len(lines1) != len(lines2):
        raise ValueError("Количество строк в файлах не совпадает")

    # Инициализация списков для хранения WER и CER
    wer_scores = []
    cer_scores = []

    # Подсчет WER и CER для каждой строки
    for line1, line2 in zip(lines1, lines2):
        line1 = line1.strip()
        line2 = line2.strip()

        # Подсчет WER
        wer = jiwer.wer(line1, line2)
        wer_scores.append(wer)

        # Подсчет CER
        cer = jiwer.cer(line1, line2)
        cer_scores.append(cer)

        # Вывод результатов для каждой строки
        print(f"Строка 1: {line1}")
        print(f"Строка 2: {line2}")
        print(f"CER: {cer:.2f}")
        print(f"WER: {wer:.2f}")
        print("-" * 30)

    return wer_scores, cer_scores


# Пример использования
file1_path = 'original.txt'
file2_path = 'model.txt'
wer_scores, cer_scores = calculate_wer_cer(file1_path, file2_path)

# Вывод средних значений WER и CER
average_wer = sum(wer_scores) / len(wer_scores)
average_cer = sum(cer_scores) / len(cer_scores)
print(f"Средний WER: {average_wer:.2f}")
print(f"Средний CER: {average_cer:.2f}")'''





