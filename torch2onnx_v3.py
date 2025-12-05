import torch
import argparse
import os
from model.vballnet_v3 import VballNetV3  # Убедитесь, что путь к модулю корректен


def export_model_to_onnx(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем чекпоинт
    checkpoint = torch.load(model_path, map_location=device)

    # Проверяем, является ли это полным чекпоинтом
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Загружены веса из 'model_state_dict' чекпоинта")
    else:
        state_dict = checkpoint
        print("✅ Прямая загрузка state_dict")

    # Инициализируем модель
    height, width, in_dim, out_dim = 288, 512, 9, 9
    model = VballNetV3(height=height, width=width, in_dim=in_dim, out_dim=out_dim)
    model.to(device)
    model.eval()

    # Загружаем state_dict
    model.load_state_dict(state_dict)
    print("✅ Веса успешно загружены в модель")

    # Дальше — как и раньше
    dummy_input = torch.randn(1, in_dim, height, width, device=device)
    onnx_path = model_path.replace(".pth", ".onnx")

    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )

    print(f"✅ Модель экспортирована в ONNX: {onnx_path}")

    # Проверка ONNX (опционально)
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX-модель валидна")
    except ImportError:
        print("ℹ️  Установите onnx: pip install onnx")
    except Exception as e:
        print(f"❌ Ошибка валидации ONNX: {e}")  


def ___export_model_to_onnx(model_path):
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем архитектуру модели
    # Параметры по умолчанию (можно изменить, если использовались другие)
    height, width, in_dim, out_dim = 288, 512, 9, 9

    model = VballNetV3(height=height, width=width, in_dim=in_dim, out_dim=out_dim)
    model.to(device)
    model.eval()  # Переводим в режим инференса

    # Загружаем веса
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Пример входного тензора
    batch_size = 1  # Для ONNX обычно фиксированный batch_size или динамический
    dummy_input = torch.randn(batch_size, in_dim, height, width, device=device)

    # Имя выходного файла
    onnx_path = model_path.replace(".pth", ".onnx")

    # Динамические оси: поддержка переменного батча и временных кадров
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    # Экспорт в ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,  # сохранить веса
            opset_version=13,  # совместимо с большинством сред
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False,
            # Убедимся, что все операции поддерживаются в ONNX
            keep_initializers_as_inputs=False
        )

    print(f"✅ Модель успешно экспортирована в ONNX: {onnx_path}")

    # Проверка корректности ONNX-модели (опционально)
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX-модель прошла проверку корректности.")
    except ImportError:
        print("ℹ️  Установите `onnx` для проверки модели: pip install onnx")
    except onnx.checker.ValidationError as e:
        print(f"❌ Ошибка проверки ONNX: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Экспорт VballNetV3 модели в ONNX")
    parser.add_argument('--model_path', type=str, required=True, help='Путь к .pth файлу модели')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Модель не найдена: {args.model_path}")

    export_model_to_onnx(args.model_path)
