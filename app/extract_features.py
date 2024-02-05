#extract_features.py
import lief
import numpy as np
import os
import csv
import joblib

properties = [
    'has_configuration',
    'has_debug',
    'has_exceptions',
    'has_exports',
    'has_imports',
    'has_nx',
    'has_relocations',
    'has_resources',
    'has_rich_header',
    'has_tls'
]
libraries = [
"*invalid*",
"ace",
"advapi32",
"api-ms-win-core-com-l1-1-0",
"api-ms-win-core-com-l1-1-1",
"api-ms-win-core-com-midlproxystub-l1-1-0",
"api-ms-win-core-debug-l1-1-0",
"api-ms-win-core-delayload-l1-1-1",
"api-ms-win-core-errorhandling-l1-1-0",
"api-ms-win-core-errorhandling-l1-1-1",
"api-ms-win-core-file-l1-1-0",
"api-ms-win-core-file-l1-2-1",
"api-ms-win-core-handle-l1-1-0",
"api-ms-win-core-heap-l1-1-0",
"api-ms-win-core-heap-l1-2-0",
"api-ms-win-core-heap-l2-1-0",
"api-ms-win-core-libraryloader-l1-2-0",
"api-ms-win-core-localization-l1-2-0",
"api-ms-win-core-localization-l1-2-1",
"api-ms-win-core-localregistry-l1-1-0",
"api-ms-win-core-memory-l1-1-2",
"api-ms-win-core-processthreads-l1-1-0",
"api-ms-win-core-processthreads-l1-1-2",
"api-ms-win-core-profile-l1-1-0",
"api-ms-win-core-registry-l1-1-0",
"api-ms-win-core-rtlsupport-l1-1-0",
"api-ms-win-core-rtlsupport-l1-2-0",
"api-ms-win-core-string-l1-1-0",
"api-ms-win-core-synch-l1-1-0",
"api-ms-win-core-synch-l1-2-0",
"api-ms-win-core-sysinfo-l1-1-0",
"api-ms-win-core-sysinfo-l1-2-1",
"api-ms-win-core-threadpool-l1-2-0",
"api-ms-win-core-winrt-error-l1-1-1",
"api-ms-win-core-winrt-string-l1-1-0",
"api-ms-win-crt-heap-l1-1-0",
"api-ms-win-crt-private-l1-1-0",
"api-ms-win-crt-runtime-l1-1-0",
"api-ms-win-crt-stdio-l1-1-0",
"api-ms-win-crt-string-l1-1-0",
"api-ms-win-downlevel-advapi32-l1-1-0",
"api-ms-win-downlevel-kernel32-l1-1-0",
"api-ms-win-downlevel-shlwapi-l1-1-0",
"api-ms-win-eventing-classicprovider-l1-1-0",
"api-ms-win-eventing-provider-l1-1-0",
"api-ms-win-security-base-l1-1-0",
"api-ms-win-security-base-l1-2-0",
"bcrypt",
"ccmcore",
"comctl32",
"comdlg32",
"crtdll",
"crypt32",
"cvbasiclib",
"dbghelp",
"dui70",
"esent",
"gdi32",
"gdiplus",
"glib-2",
"imm32",
"iphlpapi",
"kernel32",
"libgimp-2",
"libglib-2",
"libgobject-2",
"libgtk-win32-2",
"libnbbase",
"mpr",
"msasn1",
"mscoree",
"msvbvm60",
"msvcp100",
"msvcp110",
"msvcp110_win",
"msvcp120",
"msvcp140",
"msvcp60",
"msvcp71",
"msvcp80",
"msvcp90",
"msvcp_win",
"msvcr100",
"msvcr110",
"msvcr120",
"msvcr120_clr0400",
"msvcr71",
"msvcr80",
"msvcr90",
"msvcrt",
"msys-2",
"netapi32",
"ntdll",
"ntoskrnl",
"ole32",
"oleacc",
"oleaut32",
"opengl32",
"powrprof",
"psapi",
"qt5core",
"qt5gui",
"qtcore4",
"qtgui4",
"rpcrt4",
"secur32",
"setupapi",
"shell32",
"shlwapi",
"stlport",
"ulib",
"unbcl",
"urlmon",
"user32",
"userenv",
"uxtheme",
"vcruntime140",
"version",
"vmacore",
"wbemcomn",
"wex",
"wincorlib",
"winhttp",
"wininet",
"winmm",
"winspool",
"wintrust",
"ws2_32",
"wsock32",
"wtsapi32",
]

def calculate_byte_histogram(file_path, chunk_size=8192):
    byte_histogram = np.zeros(256, dtype=np.int64)
    total_bytes = 0

    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            byte_histogram += np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)

    return byte_histogram / total_bytes if total_bytes > 0 else np.zeros(256)

# Function to encode libraries from a PE file
def encode_libraries(pe):
    imports = {dll.name.lower(): [api.name if not api.is_ordinal else api.iat_address \
                                 for api in dll.entries] for dll in pe.imports}

    libs = np.zeros(len(libraries))
    for idx, lib in enumerate(libraries):
        dll_name = f"{lib}.dll"
        if lib in imports:
            libs[idx] = len(imports[lib])
        elif dll_name in imports:
            libs[idx] = len(imports[dll_name])

    total_calls = libs.sum()
    return libs / total_calls if total_calls > 0 else libs  

# Function to encode PE file
def encode_pe_file(file_path):
    try:
        pe = lief.parse(file_path)
    except Exception as e:
        print(f"An error occurred while parsing file {file_path}: {e}")
        return None

    features = []

    # Check and encode properties
    for prop in properties:
        if hasattr(pe, prop):
            features.append(float(getattr(pe, prop)))
        else:
            print(f"Attribute '{prop}' not found in {file_path}")
            features.append(0.0)

    # Encode entry point
    try:
        entrypoint = pe.optional_header.addressof_entrypoint
        raw = list(pe.get_content_from_virtual_address(entrypoint, 64))
        features.extend(raw + [0] * (64 - len(raw)))  # pad if less than 64 bytes
    except Exception as e:
        print(f"Error processing entry point for {file_path}: {e}")
        features.extend([0.0] * 64)

    # Encode byte histogram
    with open(file_path, 'rb') as file:
        file_content = file.read()
    byte_histogram = calculate_byte_histogram(file_path)
    features.extend(byte_histogram)

    # Encode imported libraries
    library_features = encode_libraries(pe)
    features.extend(library_features)

    # Encode sections information
    sections = pe.sections
    features.extend([len(sections), 
                     sum(s.entropy for s in sections) / len(sections),
                     sum(s.size for s in sections) / len(sections),
                     sum(s.virtual_size for s in sections) / len(sections)])

    # Virtual size ratio
    features.append(pe.virtual_size / max(os.path.getsize(file_path), 1))

    return np.array(features)

# Function to load the model
def load_model(model_path):
    return joblib.load(model_path)

# Function to predict using the model and processed file
def predict(model, file_path):
    features = encode_pe_file(file_path)
    if features is not None:
        features = np.array([features])  # Reshape for single sample prediction
        return model.predict(features)[0]  # Return the prediction
    else:
        return "Error processing file"

def get_feature_names():
    feature_names = properties + \
                    [f'byte_{i}' for i in range(256)] + \
                    libraries + \
                    [f'entry_raw_{i}' for i in range(64)] + \
                    ['sections_count', 'sections_entropy', 'sections_size', 'sections_virtual_size', 'virtual_size_ratio']
    return feature_names