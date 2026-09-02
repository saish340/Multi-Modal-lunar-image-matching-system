import struct, re

# --- TMC BigTIFF georeferencing ---
f = open('ch2_tmc_ndn_20231101T0125121377_d_oth_d18.tif', 'rb')
def rd(off, n):
    f.seek(off); return f.read(n)

ps_off = 2720134
tp_off = 2720158
gk_off = 2720206
gd_off = 2720382
ga_off = 2720446

ps = struct.unpack('<3d', rd(ps_off, 24))
tp = struct.unpack('<6d', rd(tp_off, 48))
gk = struct.unpack('<88H', rd(gk_off, 176))
gd = struct.unpack('<8d', rd(gd_off, 64))
ga = rd(ga_off, 129).decode('ascii', errors='replace').strip('|')

print("=== TMC GeoTIFF georeferencing ===")
print("ModelPixelScale (x, y, z):", ps)
print("ModelTiepoint (i,j,k, X,Y,Z):", tp)
print("GeoKeyDirectory (first 40 shorts):", gk[:40])
print("GeoDoubleParams:", gd)
print("GeoAsciiParams:", ga)

# Decode some key GeoKeys
GK = {gk[i]: (gk[i+1], gk[i+2], gk[i+3]) for i in range(0, len(gk)-3, 4)}
print("\n=== Key GeoKeys ===")
if 1 in GK:  # GTModelType
    print("GTModelType:", GK[1][2])
if 2 in GK:  # GSTransform
    print("GSTransform:", GK[2][2])
if 3 in GK:  # GASpheroid
    print("GASpheroid:", GK[3][2])
if 4 in GK:  # GATdatum
    print("GATdatum:", GK[4][2])
if 5 in GK:  # GADatum
    print("GADatum:", GK[5][2])
if 6 in GK:  # GEllipsoid
    print("GEllipsoid:", GK[6][2])
if 33560 in GK:  # PCS
    print("ProjectedCRS:", GK[33560][2])
if 33561 in GK:  # Projection
    print("Projection:", GK[33561][2])
if 33562 in GK:  # ProjName
    print("ProjName:", GK[33562][2])
if 33563 in GK:  # NatOriginLat
    print("NatOriginLat:", gd[GK[33563][2]-1] if GK[33563][0]==33563 else GK[33563])
if 33564 in GK:  # NatOriginLong
    print("NatOriginLong:", gd[GK[33564][2]-1] if GK[33564][0]==33564 else GK[33564])
if 33565 in GK:  # FalseEasting
    print("FalseEasting:", gd[GK[33565][2]-1] if GK[33565][0]==33565 else GK[33565])
if 33566 in GK:  # FalseNorthing
    print("FalseNorthing:", gd[GK[33566][2]-1] if GK[33566][0]==33566 else GK[33566])
if 33567 in GK:  # ProjStdParallel1
    print("ProjStdParallel1:", gd[GK[33567][2]-1] if GK[33567][0]==33567 else GK[33567])
if 33954 in GK:  # PMSCoordinateSystem
    print("PMSCoordinateSystem:", GK[33954][2])
for k in sorted(GK.keys()):
    if k not in [1,2,3,4,5,6]:
        print(f"Key {k}:", GK[k])

f = open('ch2_iir_nci_20231225T1904122779_d_img_d18.xml', 'r', encoding='utf-8')
s = f.read()
f.close()
wl = re.findall(r'<center_wavelength unit="nm">([\d.]+)</center_wavelength>', s)
bw = re.findall(r'<band_width unit="nm">([\d.]+)</band_width>', s)
print("\n=== IIRS bands ===")
print("n wavelengths:", len(wl), "n band_widths:", len(bw))
print("first 3 wavelengths:", wl[:3])
print("last 3 wavelengths:", wl[-3:])
print("min/max wavelength:", min(float(x) for x in wl), max(float(x) for x in wl))
print("first 3 bandwidths:", bw[:3])
