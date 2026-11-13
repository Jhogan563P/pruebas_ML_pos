# %% [markdown]
# Segmentación geométrica de una nube industrial
# Ejecución CPU, sin modelos entrenados. El clasificador aplica reglas iniciales
# y requiere revisión visual: no hay etiquetas de referencia para medir precisión.

# %% [markdown]
# 1. Instala dependencias en Colab (si aún no están disponibles).

# %%
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open3d", "numpy", "scipy", "scikit-learn", "matplotlib"])

# %% [markdown]
# 2. Configuración. Ajusta especialmente las escalas y umbrales a tu escaneo.

# %%
from pathlib import Path
import json, time, warnings
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

CONFIG = {
    "PLY_PATH": "/content/nave.ply", "OUTPUT_DIR": "/content/resultado_segmentacion",
    "SEED": 42, "VOXEL": 0.05, "USE_FILE_NORMALS": False, "USE_RGB": False,
    "UP_AXIS": 2, "HIST_BIN": 0.05, "NORMAL_UP": 0.90,
    "FLOOR_BAND": 0.15, "CEILING_BAND": 0.30,
    "PLANE_DIST": 0.03, "PLANE_ITERS": 1000, "MAX_PLANES": 15,
    "MIN_PLANE_POINTS": 1500, "PLANE_PIECE_EPS": 0.15,
    "KNN_K": 20, "KNN_SAMPLE": 300, "ETA": 2.0, "DBSCAN_MIN": 5,
    "GIANT_FRACTION": 0.50, "GIANT_ETA_FACTOR": 0.70,
    "CYLINDER_MAX_POINTS": 5000, "CYLINDER_TRIALS": 120,
    "CYLINDER_RADIUS_MIN": 0.03, "CYLINDER_RADIUS_MAX": 0.60,
    "CYLINDER_INLIER_RATIO": 0.45, "TINY_POINTS": 25,
    # Reglas geométricas iniciales (metros); ajústalas en función de la escena.
    "WALL_PLANARITY": 0.72, "WALL_VERTICALITY": 0.75,
    "ELONGATED_LINEARITY": 0.78, "COLUMN_VERTICALITY": 0.80,
    "PERSON_HEIGHT_MIN": 1.40, "PERSON_HEIGHT_MAX": 2.10,
    "PERSON_FOOTPRINT_MAX": 1.20, "PERSON_BASE_MAX": 0.35,
    "MACHINE_FOOTPRINT_MIN": 1.50, "MACHINE_HEIGHT_MAX": 3.50,
    "MACHINE_BASE_MAX": 0.50,
    # DBSCAN recomendado. Alternativas: "kmeans" y "gmm" (sobre XYZ residual).
    "RESIDUAL_METHOD": "dbscan", "ALT_CLUSTER_COUNT": 12,
    "SMOKE_TEST": False, "SMOKE_POINTS": 100_000,
}

SEED = CONFIG["SEED"]
np.random.seed(SEED)
CLASS_NAMES = {0:"piso", 1:"techo", 2:"pared", 3:"columna", 4:"tuberia",
               5:"viga", 6:"persona", 7:"maquina_vehiculo", 8:"otro"}
COLORS = np.array([[0.25,.55,.95],[.55,.55,.60],[.90,.35,.25],[.20,.75,.35],
                   [.10,.80,.80],[.90,.70,.15],[.95,.25,.70],[.55,.30,.85],[.65,.65,.65]])
TIMINGS = []

def timed(name, count, fn):
    t = time.perf_counter(); out = fn(); dt = time.perf_counter()-t
    TIMINGS.append({"etapa":name,"segundos":dt,"puntos":int(count),"puntos_segundo":float(count/dt) if dt else 0})
    return out

def load_cloud(path):
    """Carga un PLY XYZ y obtiene opcionalmente sus atributos RGB/normales."""
    cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(cloud.points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("No se pudieron leer puntos XYZ del archivo PLY.")
    rgb = np.asarray(cloud.colors) if cloud.has_colors() else None
    normals = np.asarray(cloud.normals) if cloud.has_normals() else None
    return xyz, rgb, normals

def voxelize(xyz, voxel):
    """Submuestrea y conserva los índices originales asociados a cada vóxel."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    bounds = pc.get_axis_aligned_bounding_box()
    down, _, trace = pc.voxel_down_sample_and_trace(voxel, bounds.min_bound, bounds.max_bound, False)
    centers = np.asarray(down.points)
    original = [np.asarray(ids, dtype=np.int64) for ids in trace]
    return centers, original

def estimate_normals(xyz, provided, config):
    """Usa normales del archivo solo si está configurado; de otro modo las estima."""
    if config["USE_FILE_NORMALS"] and provided is not None and len(provided)==len(xyz):
        n = np.asarray(provided, dtype=float)
        good = np.isfinite(n).all(axis=1) & (np.linalg.norm(n,axis=1)>1e-8)
        if good.all(): return n / np.linalg.norm(n,axis=1,keepdims=True)
        warnings.warn("Normales de archivo inválidas; se estimarán desde XYZ.")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(3*config["VOXEL"],.12), max_nn=30))
    n = np.asarray(pc.normals)
    return n / np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)

def check_up_axis(xyz, normals, config):
    axis = config["UP_AXIS"]
    med = np.nanmedian(np.abs(normals),axis=0)
    if axis != 2 or med[2] < 0.35:
        warnings.warn(f"El eje UP_AXIS={axis} puede no ser vertical (mediana |normal|={med.round(2)}). Revisa el eje y rota XYZ si hace falta.")

def detect_floor_ceiling(xyz, normals, config):
    """Detecta bandas horizontales con histograma rellenado en los extremos."""
    axis=config["UP_AXIS"]; z=xyz[:,axis]; nz=np.abs(normals[:,axis])
    cand=nz>=config["NORMAL_UP"]
    if cand.sum()<max(100,int(.005*len(xyz))): cand=np.ones(len(xyz),bool)
    bw=config["HIST_BIN"]
    edges=np.arange(z.min()-2*bw,z.max()+3*bw,bw)
    hist,_=np.histogram(z[cand],bins=edges)
    smooth=np.convolve(hist,np.ones(3),mode="same")
    peaks,_=find_peaks(smooth, distance=max(1,int(.30/bw)))
    if len(peaks)<2: peaks=np.argsort(smooth)[-min(2,len(smooth)):]
    peaks=sorted(peaks,key=lambda i:smooth[i],reverse=True)
    centers=(edges[:-1]+edges[1:])/2
    # Extremos espaciales entre los picos dominantes suelen corresponder a piso/techo.
    chosen=sorted(peaks[:min(6,len(peaks))],key=lambda i:centers[i])
    floor_z=float(centers[chosen[0]]); ceil_z=float(centers[chosen[-1]])
    horizontal=nz>=config["NORMAL_UP"]
    floor=(np.abs(z-floor_z)<=config["FLOOR_BAND"]) & horizontal
    ceiling=(np.abs(z-ceil_z)<=config["CEILING_BAND"]) & horizontal & ~floor
    if floor_z>=ceil_z: warnings.warn("No se detectó piso por debajo del techo; revisar ejes y bandas.")
    return floor,ceiling,floor_z,ceil_z

def dbscan_labels(points, eps, min_samples):
    if len(points)==0: return np.empty(0,dtype=int)
    return DBSCAN(eps=float(eps),min_samples=int(min_samples),n_jobs=-1).fit_predict(points)

def extract_planes(xyz, normals, excluded, config):
    """Extrae planos RANSAC iterativos y separa cada plano en piezas conectadas."""
    remaining=np.flatnonzero(~excluded); plane_ids=np.full(len(xyz),-1,int); next_id=0
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    for _ in range(config["MAX_PLANES"]):
        if len(remaining)<config["MIN_PLANE_POINTS"]: break
        sub=cloud.select_by_index(remaining.tolist())
        model,inliers=sub.segment_plane(config["PLANE_DIST"],3,config["PLANE_ITERS"],.999)
        if len(inliers)<config["MIN_PLANE_POINTS"]: break
        ids=remaining[np.asarray(inliers,dtype=int)]
        local=dbscan_labels(xyz[ids],config["PLANE_PIECE_EPS"],max(5,config["DBSCAN_MIN"]))
        for lab in np.unique(local):
            piece=ids[local==lab]
            if len(piece)>=config["TINY_POINTS"]:
                plane_ids[piece]=next_id; next_id+=1
        mask=np.ones(len(remaining),bool); mask[np.asarray(inliers,dtype=int)]=False
        remaining=remaining[mask]
    return plane_ids

def adaptive_eps(points, config, rng):
    if len(points)<3: return config["ETA"]*config["VOXEL"]
    take=rng.choice(len(points),min(config["KNN_SAMPLE"],len(points)),replace=False)
    k=min(config["KNN_K"]+1,len(points))
    d,_=cKDTree(points).query(points[take],k=k,workers=-1)
    kth=d[:,-1] if d.ndim==2 else d
    med=float(np.median(kth[np.isfinite(kth)]))
    return max(config["VOXEL"],config["ETA"]*med)

def cluster_residual(points, normals, config, rng):
    """Agrupa residuo; los bloques gigantes se subdividen por orientación y escala."""
    method=config["RESIDUAL_METHOD"].lower()
    eps=adaptive_eps(points,config,rng)
    if method=="dbscan": labels=dbscan_labels(points,eps,config["DBSCAN_MIN"])
    elif method=="kmeans": labels=KMeans(n_clusters=min(config["ALT_CLUSTER_COUNT"],len(points)),random_state=SEED,n_init=5).fit_predict(points)
    elif method=="gmm": labels=GaussianMixture(n_components=min(config["ALT_CLUSTER_COUNT"],len(points)),random_state=SEED).fit_predict(points)
    elif method=="hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as e: raise RuntimeError("HDBSCAN no está disponible en esta versión de scikit-learn.") from e
        labels=HDBSCAN(min_cluster_size=max(10,config["DBSCAN_MIN"])).fit_predict(points)
    else: raise ValueError("RESIDUAL_METHOD debe ser dbscan, hdbscan, kmeans o gmm.")
    valid=labels>=0
    if valid.any():
        ids,counts=np.unique(labels[valid],return_counts=True); giant=ids[np.argmax(counts)]
        if counts.max()/len(points)>config["GIANT_FRACTION"]:
            gidx=np.flatnonzero(labels==giant); nz=np.abs(normals[gidx,config["UP_AXIS"]])
            fam=np.where(nz>.85,0,np.where(nz<.25,1,2))
            new=labels.copy(); next_id=int(labels.max())+1
            for family in range(3):
                ix=gidx[fam==family]
                if len(ix)<config["DBSCAN_MIN"]: continue
                sub=dbscan_labels(points[ix],eps*config["GIANT_ETA_FACTOR"],config["DBSCAN_MIN"])
                for sl in np.unique(sub):
                    if sl<0: continue
                    si=ix[sub==sl]
                    if len(si)>=config["TINY_POINTS"]: new[si]=next_id; next_id+=1
                    else: new[si]=-1
            labels=new
    return labels,eps

def cylinder_check(points,normals,config,rng):
    """Ajuste circular RANSAC en plano perpendicular al eje PCA principal."""
    n=len(points)
    if n<30: return {"ok":False,"ratio":0.,"radius":None}
    ids=rng.choice(n,min(n,config["CYLINDER_MAX_POINTS"]),replace=False)
    p=points[ids]; nn=normals[ids]
    _,_,vh=np.linalg.svd(p-p.mean(axis=0),full_matrices=False); axis=vh[0]
    ref=np.array([1.,0,0]) if abs(axis[0])<.9 else np.array([0.,1,0])
    u=np.cross(axis,ref); u/=np.linalg.norm(u); v=np.cross(axis,u)
    q=np.column_stack(((p-p.mean(axis=0))@u,(p-p.mean(axis=0))@v))
    qn=np.column_stack((nn@u,nn@v)); best=0; best_r=None
    trials=min(config["CYLINDER_TRIALS"],max(20,n//10))
    for _ in range(trials):
        ii=rng.choice(n,3,replace=False); a,b,c=q[ii]
        mat=2*np.array([b-a,c-a]); det=np.linalg.det(mat)
        if abs(det)<1e-8: continue
        center=np.linalg.solve(mat,np.array([b@b-a@a,c@c-a@a])); radius=np.linalg.norm(a-center)
        if not config["CYLINDER_RADIUS_MIN"]<=radius<=config["CYLINDER_RADIUS_MAX"]: continue
        radial=q-center; rr=np.linalg.norm(radial,axis=1)
        agree=np.abs(np.sum((radial/np.maximum(rr[:,None],1e-9))*qn,axis=1))>.65
        inside=(np.abs(rr-radius)<max(.025,config["VOXEL"]*1.5)) & agree
        count=int(inside.sum())
        if count>best: best=count; best_r=float(radius)
    ratio=best/n
    return {"ok":ratio>=config["CYLINDER_INLIER_RATIO"],"ratio":ratio,"radius":best_r}

def cluster_features(points,normals,floor_z,config,rng):
    center=points.mean(axis=0); cov=np.cov(points-center,rowvar=False) if len(points)>2 else np.eye(3)*1e-9
    eig,vec=np.linalg.eigh(cov); order=np.argsort(eig)[::-1]; eig=np.maximum(eig[order],1e-12); vec=vec[:,order]
    linear=float((eig[0]-eig[1])/eig[0]); planar=float((eig[1]-eig[2])/eig[0]); spherical=float(eig[2]/eig[0])
    extent=np.ptp(points,axis=0); up=config["UP_AXIS"]
    other=[i for i in range(3) if i!=up]; footprint=float(extent[other[0]]*extent[other[1]])
    base=float(points[:,up].min()-floor_z); height=float(extent[up]); vertical=float(abs(vec[up,0]))
    elongated=linear>=config["ELONGATED_LINEARITY"]
    cyl=cylinder_check(points,normals,config,rng) if elongated else {"ok":False,"ratio":0.,"radius":None}
    return {"linearidad":linear,"planaridad":planar,"esfericidad":spherical,"verticalidad":vertical,
            "extensiones":extent,"huella":footprint,"altura":height,"base_piso":base,
            "elongado":elongated,"cilindro":cyl}

def classify_features(f,config):
    if f["planaridad"]>=config["WALL_PLANARITY"] and f["verticalidad"]>=config["WALL_VERTICALITY"]: return 2,.72
    if f["elongado"] and f["verticalidad"]>=config["COLUMN_VERTICALITY"]: return 3,.68
    if f["elongado"] and f["cilindro"]["ok"] and f["verticalidad"]<config["COLUMN_VERTICALITY"]: return 4,min(.95,.5+f["cilindro"]["ratio"])
    if f["elongado"]: return 5,.60
    if config["PERSON_HEIGHT_MIN"]<=f["altura"]<=config["PERSON_HEIGHT_MAX"] and f["huella"]<=config["PERSON_FOOTPRINT_MAX"] and f["base_piso"]<=config["PERSON_BASE_MAX"]: return 6,.55
    if f["huella"]>=config["MACHINE_FOOTPRINT_MIN"] and f["altura"]<=config["MACHINE_HEIGHT_MAX"] and f["base_piso"]<=config["MACHINE_BASE_MAX"]: return 7,.52
    return 8,.35

def predict(xyz,rgb=None,normals=None,config=None):
    """Devuelve etiquetas/clases, score heurístico e ID de instancia por punto original."""
    cfg=dict(CONFIG); cfg.update(config or {}); rng=np.random.default_rng(cfg["SEED"])
    xyz=np.asarray(xyz,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3: raise ValueError("xyz debe tener forma (N,3).")
    started=time.perf_counter(); vox,mapping=timed("0_voxel",len(xyz),lambda:voxelize(xyz,cfg["VOXEL"]))
    vn=timed("normales",len(vox),lambda:estimate_normals(vox,None,cfg))
    check_up_axis(vox,vn,cfg)
    floor,ceiling,fz,cz=timed("1_piso_techo",len(vox),lambda:detect_floor_ceiling(vox,vn,cfg))
    excluded=floor|ceiling
    planes=timed("2_planos_ransac",int((~excluded).sum()),lambda:extract_planes(vox,vn,excluded,cfg))
    vlabels=np.full(len(vox),-1,int); scores=np.zeros(len(vox),float); instance=np.full(len(vox),-1,int)
    vlabels[floor]=0; scores[floor]=.9; vlabels[ceiling]=1; scores[ceiling]=.9
    next_inst=0
    for pid in np.unique(planes[planes>=0]):
        ix=np.flatnonzero(planes==pid)
        # Los planos verticales se etiquetan como pared; superficies planas restantes
        # quedan como "otro" para no confundir mesas o plataformas con paredes.
        vertical_plane=float(np.median(np.abs(vn[ix,cfg["UP_AXIS"]])))<.30
        vlabels[ix]=2 if vertical_plane else 8; scores[ix]=.72 if vertical_plane else .45
        instance[ix]=next_inst; next_inst+=1
    residual=np.flatnonzero((~excluded)&(planes<0))
    labs,eps=timed("3_residuo_cluster",len(residual),lambda:cluster_residual(vox[residual],vn[residual],cfg,rng))
    cluster_labels,_=labs
    clusters=np.unique(cluster_labels[cluster_labels>=0])
    for lab in clusters:
        loc=np.flatnonzero(cluster_labels==lab); ix=residual[loc]
        if len(ix)<cfg["TINY_POINTS"]: continue
        f=cluster_features(vox[ix],vn[ix],fz,cfg,rng); cl,sc=classify_features(f,cfg)
        vlabels[ix]=cl; scores[ix]=sc; instance[ix]=next_inst; next_inst+=1
    # Propaga a todos los originales por traza de vóxel, y usa vecino para puntos no trazados.
    outlab=np.full(len(xyz),-1,int); outscore=np.zeros(len(xyz)); outinst=np.full(len(xyz),-1,int)
    mapped=np.zeros(len(xyz),bool)
    for vi,orig in enumerate(mapping):
        orig=orig[(orig>=0)&(orig<len(xyz))]
        outlab[orig]=vlabels[vi]; outscore[orig]=scores[vi]; outinst[orig]=instance[vi]; mapped[orig]=True
    missing=np.flatnonzero(~mapped)
    if len(missing):
        _,nn=cKDTree(vox).query(xyz[missing],k=1,workers=-1)
        outlab[missing]=vlabels[nn]; outscore[missing]=scores[nn]; outinst[missing]=instance[nn]
    TIMINGS.append({"etapa":"total","segundos":time.perf_counter()-started,"puntos":len(xyz),"puntos_segundo":len(xyz)/max(time.perf_counter()-started,1e-9)})
    return {"labels":outlab,"scores":outscore,"instance_id":outinst,"floor_z":fz,"ceiling_z":cz,"voxel_xyz":vox,"voxel_labels":vlabels,"eps":eps}

def evaluate(labels,gt_labels):
    """Métricas supervisadas por clase para cuando se disponga de etiquetas manuales."""
    y=np.asarray(gt_labels); p=np.asarray(labels)
    if y.shape!=p.shape: raise ValueError("labels y gt_labels deben tener la misma forma.")
    result={}
    for c in np.unique(y):
        tp=int(np.sum((p==c)&(y==c))); fp=int(np.sum((p==c)&(y!=c))); fn=int(np.sum((p!=c)&(y==c)))
        result[int(c)]={"precision":tp/(tp+fp) if tp+fp else 0.,"recall":tp/(tp+fn) if tp+fn else 0.,"IoU":tp/(tp+fp+fn) if tp+fp+fn else 0.}
    return result

def self_consistency(xyz,labels,config):
    rng=np.random.default_rng(config["SEED"]); n=min(50000,len(xyz)); ix=rng.choice(len(xyz),n,replace=False)
    k=min(11,len(xyz)); _,nn=cKDTree(xyz).query(xyz[ix],k=k,workers=-1)
    neigh=labels[nn[:,1:]]; own=labels[ix]
    disagree=np.mean(np.mean(neigh!=own[:,None],axis=1)>.5) if neigh.size else 0.
    sizes=np.unique(labels[labels>=0],return_counts=True)[1]
    return {"porcentaje_vecinos_mayoria_otra_clase":float(disagree*100),"porcentaje_sin_asignar":float(np.mean(labels<0)*100),
            "segmentos_pequenos":int(np.sum(sizes<config["TINY_POINTS"])),"floor_below_ceiling":bool(RESULT["floor_z"]<RESULT["ceiling_z"]),
            "floor_far_above_level":bool(np.any((labels==0)&(xyz[:,config["UP_AXIS"]]>RESULT["floor_z"]+config["FLOOR_BAND"]*2)))}

def export_results(xyz,result,config):
    out=Path(config["OUTPUT_DIR"]); out.mkdir(parents=True,exist_ok=True)
    labels=result["labels"]; inst=result["instance_id"]
    color=np.tile(np.array([.2,.2,.2]),(len(xyz),1))
    for c,col in enumerate(COLORS): color[labels==c]=col
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)); pc.colors=o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(str(out/"ply_by_class.ply"),pc,write_ascii=False)
    rng=np.random.default_rng(config["SEED"]); ids=np.unique(inst[inst>=0]); palette=rng.uniform(.15,.95,(max(1,len(ids)),3)); imap={v:i for i,v in enumerate(ids)}
    ci=np.tile(np.array([.15,.15,.15]),(len(xyz),1))
    for v,i in imap.items(): ci[inst==v]=palette[i]
    ip=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)); ip.colors=o3d.utility.Vector3dVector(ci)
    o3d.io.write_point_cloud(str(out/"ply_by_instance.ply"),ip,write_ascii=False)
    np.savez_compressed(out/"labels.npz",labels=labels,scores=result["scores"],instance_id=inst)
    counts={CLASS_NAMES[c]:int(np.sum(labels==c)) for c in CLASS_NAMES}
    stats={"points":len(xyz),"points_per_class":counts,"percent_unassigned":float(np.mean(labels<0)*100),
           "cluster_size_distribution":{"min":int(np.min(np.unique(inst[inst>=0],return_counts=True)[1])) if np.any(inst>=0) else 0,
             "median":float(np.median(np.unique(inst[inst>=0],return_counts=True)[1])) if np.any(inst>=0) else 0,
             "max":int(np.max(np.unique(inst[inst>=0],return_counts=True)[1])) if np.any(inst>=0) else 0},
           "tiny_segments":int(np.sum(np.unique(inst[inst>=0],return_counts=True)[1]<config["TINY_POINTS"])) if np.any(inst>=0) else 0,
           "floor_z":result["floor_z"],"ceiling_z":result["ceiling_z"],"adaptive_eps":result["eps"],"self_consistency":self_consistency(xyz,labels,config),"timings":TIMINGS,
           "validation":"Reglas geométricas; no validado contra ground truth."}
    (out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    # Vistas ortográficas y leyenda.
    for axis,name in ((1,"top"),(2,"side")):
        keep=np.arange(len(xyz));
        if len(keep)>200000: keep=np.random.default_rng(config["SEED"]).choice(keep,200000,replace=False)
        horiz=[i for i in range(3) if i!=axis]
        fig,ax=plt.subplots(figsize=(11,7)); ax.scatter(xyz[keep,horiz[0]],xyz[keep,horiz[1]],c=color[keep],s=.15,rasterized=True)
        ax.set_aspect("equal",adjustable="box"); ax.set_title(f"Vista {name}: clases heurísticas"); ax.set_xlabel("m"); ax.set_ylabel("m")
        fig.savefig(out/f"preview_{name}.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.axis("off")
    ax.legend([plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS[c],markersize=8) for c in CLASS_NAMES],
              [CLASS_NAMES[c] for c in CLASS_NAMES],loc="center",ncol=3); fig.savefig(out/"leyenda.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    return out,stats

# %% [markdown]
# 3. En Colab, carga primero el PLY desde el panel de archivos y actualiza PLY_PATH.

# %%
if CONFIG["SMOKE_TEST"]:
    if not Path(CONFIG["PLY_PATH"]).exists():
        _candidates=list(Path("/content").glob("*.ply"))
        if len(_candidates)==1: CONFIG["PLY_PATH"]=str(_candidates[0])
    _xyz,_rgb,_normals=load_cloud(CONFIG["PLY_PATH"])
    _rng=np.random.default_rng(CONFIG["SEED"])
    _take=_rng.choice(len(_xyz),min(CONFIG["SMOKE_POINTS"],len(_xyz)),replace=False)
    RESULT=predict(_xyz[_take],None if _rgb is None else _rgb[_take],None if _normals is None else _normals[_take])
    print("Smoke test ejecutado con",len(_take),"puntos. Para procesar todo, cambia SMOKE_TEST=False y vuelve a ejecutar.")
else:
    if not Path(CONFIG["PLY_PATH"]).exists():
        _candidates=list(Path("/content").glob("*.ply"))
        if len(_candidates)==1: CONFIG["PLY_PATH"]=str(_candidates[0])
    _xyz,_rgb,_normals=timed("carga",0,lambda:load_cloud(CONFIG["PLY_PATH"]))
    # Las normales se recalculan en los vóxeles por defecto; RGB se ignora salvo activación explícita.
    RESULT=predict(_xyz,_rgb if CONFIG["USE_RGB"] else None,_normals)
    OUT,STATS=export_results(_xyz,RESULT,CONFIG)
    print("Resultados:",OUT)
    print("Clasificación basada en reglas; no validada contra ground truth.")
    print("Piso:",round(RESULT["floor_z"],3),"| techo:",round(RESULT["ceiling_z"],3),"| eps DBSCAN:",round(RESULT["eps"],3))
    print("Etapa                         segundos       puntos/s")
    for row in TIMINGS: print(f"{row['etapa']:<28} {row['segundos']:>8.2f} {row['puntos_segundo']:>14,.0f}")
    try:
        import resource
        _peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _peak_gb=_peak/1e6 if _peak>10**7 else _peak/1e9
        print(f"RAM pico del proceso: {_peak_gb:.2f} GB (estimación del sistema operativo)")
    except (ImportError,AttributeError): print("RAM pico: no disponible en esta plataforma.")
    print("Ajusta VOXEL, ETA/DBSCAN_MIN, bandas de piso/techo y umbrales geométricos; reduce ETA si reaparece un cluster gigante.")
