import unreal
import random,glob

def buildMeshImportOptions():
    options = unreal.FbxImportUI()
    skeleton = unreal.EditorAssetLibrary.load_asset('/Game/makehuman/anim/Walking_Skeleton')
    options.set_editor_property('import_as_skeletal',True)
    options.set_editor_property('import_materials',True)
    options.set_editor_property('import_mesh',True)
    options.set_editor_property('skeleton',skeleton)
    return options

def texturefactory():
    f = unreal.TextureFactory()
    f.set_editor_property('create_material',False)
    return f

def importTextureTask(filename,destination_path="/Game/mh_models/textures_ccp"):
    task = unreal.AssetImportTask()
    task.set_editor_property('automated', True)
    task.set_editor_property('destination_path', destination_path)
    task.set_editor_property('filename', filename)
    task.set_editor_property('replace_existing', True)
    task.set_editor_property('save', True)
    task.set_editor_property('factory', texturefactory())
    return task

def importFBXTask(filename,destination_path='/Game/MHModel/SklMeshView'):
    task= unreal.AssetImportTask()
    task.set_editor_property('automated',True)
    task.set_editor_property('destination_path',destination_path)
    task.set_editor_property('filename',filename)
    task.set_editor_property('replace_existing',True)
    task.set_editor_property('save',True)
    task.set_editor_property('options', buildMeshImportOptions())
    return task

def executeImportTask(tasks):
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    for task in tasks:
        print("File:" + task.get_editor_property('filename'))
        for path in task.get_editor_property('imported_object_paths'):
            print("Imported: "+ path)

def  mass_import_textures(start,end):
    #files = glob.glob("/Users/zhangtianyu/Downloads/val2017/*.jpg")
    files= glob.glob('/Users/zhangtianyu/Downloads/clothing-co-parsing-master/ccp_patches/*.png')
    files.sort()
    tasks = [importTextureTask(f) for f in files[start:end]]
    executeImportTask(tasks)

def mass_import_fbx(start,end):
    files = ['f:/mhview/mhview_{:04d}A.fbx'.format(i) for i in range(start,end+1)]
    tasks = [importFBXTask(f) for f in files]
    executeImportTask(tasks)

def spawnActor(tps,x,mesh):
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHMasterAI')
    loc = unreal.Vector(random.uniform(-1500,-1000),random.uniform(1200,7000),129)
    rotation = unreal.Rotator(0,0,random.uniform(0,360))
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class,loc,rotation)
    actor.set_editor_property("Path",tps)

    actor.mesh.set_editor_property("skeletal_mesh",mesh)
    actor.set_actor_label(x)
    return actor


def executeSpawnTask(start=1,end=1001):
    # laod skeletal mesh
    rpc=['uplow{:04d}A'.format(i) for i in range(start,end)]
    mesh_list = []
    for i in range(len(rpc)):
        mesh = unreal.EditorAssetLibrary.load_asset("/Game/mh_models/"+rpc[i])
        mesh_list.append(mesh)

    quantity = len(rpc)
    tp_list=[]
    allactors = unreal.EditorLevelLibrary.get_all_level_actors()
    for a in allactors:
        if type(a)==unreal.TargetPoint:
            tp_list.append(a)
    print(tp_list)
    with unreal.ScopedSlowTask(quantity,"actor spawning") as sst:
        sst.make_dialog(True)


        for x  in range(quantity):
            tps = unreal.Array(unreal.TargetPoint)
            t = random.sample(tp_list, 5)
            for tcount in range(4,10):
                tps.append(tp_list[tcount])
            for tcount in range(0,4):
                tps.append(tp_list[tcount])
            if sst.should_cancel():
                break
            sst.enter_progress_frame(1)
            spawnActor(tps,rpc[x],mesh_list[x])

def changeRoughnessToOne(start,end):
    rpc = ['uplow{:04d}A'.format(i) for i in range(start, end)]
    with unreal.ScopedSlowTask(1000, "actor spawning") as sst:
        for i in unreal.EditorUtilityLibrary.get_selected_assets():


            mat=i
            ts_TextureNodeBc = unreal.MaterialEditingLibrary.create_material_expression(mat,unreal.MaterialExpressionConstant,0,0)
            ts_TextureNodeBc.r=1
            unreal.MaterialEditingLibrary.connect_material_property(ts_TextureNodeBc,"",unreal.MaterialProperty.MP_ROUGHNESS)
            sst.enter_progress_frame(1)

def ana_mesh_materials():
    rpc = ['mhview_{:04d}A'.format(i) for i in range(1, 800)]
    mesh_mat=dict()
    for i in range(len(rpc)):
        mesh = unreal.EditorAssetLibrary.load_asset("/Game/MHModel/SklMeshView/" + rpc[i])
        if mesh is None:break
        materials = mesh.materials
        names=[]
        if len(materials) != 16:
            print("{} {}".format(rpc[i],len(materials)))

def del_ana_mesh_materials():
    rpc = ['mhmodel_view_{:04d}A'.format(i) for i in range(1, 800)]
    mesh_mat=dict()
    for i in range(len(rpc)):
        mesh = unreal.EditorAssetLibrary.load_asset("/Game/MHModel/SklMeshView/" + rpc[i])
        if mesh is None:break
        materials = mesh.materials
        names=[]
        if len(materials) != 16:
            print("{} {}".format(rpc[i],len(materials)))
            unreal.EditorAssetLibrary.delete_loaded_asset(mesh)



def get_mesh_materials():
    rpc = ['uplow{:04d}A'.format(i) for i in range(1, 1801)]
    mesh_mat=dict()
    for i in range(1000,1800):
        mesh = unreal.EditorAssetLibrary.load_asset("/Game/mh_models/" + rpc[i])
        materials = mesh.materials
        names=[]
        for m in materials:
            name = str(m.material_slot_name)
            names.append(name)
        mesh_mat[rpc[i]]=names

    import json
    json.dump(mesh_mat,open("/Users/zhangtianyu/jsonout/meshmat1800.json",'w'))

def set_shading_model():
    with unreal.ScopedSlowTask(1000, " msm changing") as sst:
        for i in unreal.EditorUtilityLibrary.get_selected_assets():
            m=i
            m.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_DEFAULT_LIT)
            sst.enter_progress_frame(1)

def set_mhbp_randtexture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHMasterAI')
    cdo = unreal.get_default_object(actor_class)

    mats = unreal.EditorUtilityLibrary.get_selected_assets()
    mat_a = unreal.Array(unreal.SkeletalMesh)
    for m in mats:
        mat_a.append(m)
    cdo.set_editor_property("mesh_list", mat_a)

def set_mhbp_mesh_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClassDivide')
    cdo = unreal.get_default_object(actor_class)

    mats = unreal.EditorUtilityLibrary.get_selected_assets()
    mat_a = unreal.Array(unreal.SkeletalMesh)
    for m in mats:
        mat_a.append(m)
    cdo.set_editor_property("mesh_list", mat_a)

def set_top_texture_list_for75():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClassSameClothes75')
    cdo = unreal.get_default_object(actor_class)
    import glob , random
    files = glob.glob("F:\\75_clothes_textures\\*.png")


    mat_a = unreal.Array(str)
    mat_a.extend(files)
    print(files[:10])
    cdo.set_editor_property("clo_texture", mat_a)

def set_top_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClass')
    cdo = unreal.get_default_object(actor_class)
    import glob , random
    files = glob.glob("F:\\df_patches\\top*.png")+glob.glob("F:\\df_patches\\outer*.png")
    files = random.sample(files,2000)
    tops = ['jacket','blouse','coat','sweater','blazer','cardigan','t-shirt','shirt','suit','sweatshirt',
            'vest','jumper','bodysuit','top','romper']
    for t in tops:
        files.extend(glob.glob("F:\\ccp_patches\\{}*.png".format(t)))

    mat_a = unreal.Array(str)
    mat_a.extend(files)
    print(files[:10])
    cdo.set_editor_property("top_texture", mat_a)

def set_rnd_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClass')
    cdo = unreal.get_default_object(actor_class)
    import glob,random
    files = glob.glob("F:\\ccp_patches\\*.png")+glob.glob("F:\\df_patches\\*.png")
    files = random.sample(files,100)


    mat_a = unreal.Array(str)
    mat_a.extend(files)
    cdo.set_editor_property("rnd_texture", mat_a)

def set_bag_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClass')
    cdo = unreal.get_default_object(actor_class)
    import glob,random
    files = []
    downs = ['bag','purse']
    for t in downs:
        files.extend(glob.glob("F:\\ccp_patches\\{}*.png".format(t)))

    mat_a = unreal.Array(str)
    mat_a.extend(files)
    cdo.set_editor_property("bag_texture", mat_a)

def set_scarf_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClass')
    cdo = unreal.get_default_object(actor_class)
    import glob,random
    files = []
    downs = ['scarf']
    for t in downs:
        files.extend(glob.glob("F:\\ccp_patches\\{}*.png".format(t)))

    mat_a = unreal.Array(str)
    mat_a.extend(files)
    cdo.set_editor_property("scarf_texture", mat_a)

def set_black_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClassBL')
    cdo = unreal.get_default_object(actor_class)
    import glob

    import glob, random
    files = glob.glob("F:\\df_patches\\top*.png") + glob.glob("F:\\df_patches\\outer*.png")
    files = random.sample(files, 2000)
    tops = ['jacket', 'blouse', 'coat', 'sweater', 'blazer', 'cardigan', 't-shirt', 'shirt', 'suit', 'sweatshirt',
            'vest', 'jumper', 'bodysuit', 'top', 'romper']
    for t in tops:
        files.extend(glob.glob("F:\\ccp_patches\\{}*.png".format(t)))
    files.extend(glob.glob("F:\\black_patches_expand\\*g"))
    mat_a = unreal.Array(str)
    mat_a.extend(files)
    cdo.set_editor_property("topblack_texture", mat_a)

    files = []
    files = glob.glob("F:\\df_patches\\pants*.png")
    files = random.sample(files,2000)
    files.extend(glob.glob("F:/df_patches/skirt*.png")+glob.glob("F:\\df_patches\\dress*.png"))
    downs = ['pants','jeans','shorts','skirt','leggings','tights','stockings']
    for t in downs:
        files.extend(glob.glob("F:\\ccp_patches\\{}*.png".format(t)))
    files.extend(glob.glob("F:\\black_patches_expand\\*g"))
    mat_b = unreal.Array(str)
    mat_b.extend(files)
    cdo.set_editor_property("downblack_texture", mat_b)


def set_rp_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClassRP')
    cdo = unreal.get_default_object(actor_class)
    import glob
    files = glob.glob("F:\\pattern\\pattern\\*g")
    mat_a = unreal.Array(str)
    mat_a.extend(files)
    cdo.set_editor_property("black_texture", mat_a)


def set_down_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClass')
    cdo = unreal.get_default_object(actor_class)
    import glob,random
    files = glob.glob("F:\\df_patches\\pants*.png")
    files = random.sample(files,2000)
    files.extend(glob.glob("F:/df_patches/skirt*.png")+glob.glob("F:\\df_patches\\dress*.png"))
    downs = ['pants','jeans','shorts','skirt','leggings','tights','stockings']
    for t in downs:
        files.extend(glob.glob("F:\\ccp_patches\\{}*.png".format(t)))

    mat_a = unreal.Array(str)
    mat_a.extend(files)
    cdo.set_editor_property("down_texture", mat_a)

def set_ccp_texture_list():
    actor_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MHModel/MHMainClassDivide')
    cdo = unreal.get_default_object(actor_class)
    import glob,random
    files = glob.glob("F:\\ccp_patches\\*.png")

    mat_a = unreal.Array(str)
    mat_a.extend(files)
    cdo.set_editor_property("ccp_texture", mat_a)


if __name__=='__main__':
    executeSpawnTask()

