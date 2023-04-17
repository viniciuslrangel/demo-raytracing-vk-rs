#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_albedo;
layout(location = 2) out vec3 f_normal;
layout(location = 3) out float f_depth;

layout(location = 0) in vec2 coord;

layout(set = 0, binding = 0) uniform ViewData {
    mat4 worldview;
    mat4 proj;
    float blur;
} viewData;

layout(set = 0, binding = 1) uniform RenderInfo {
    float time;
    int sample_count;
} renderInfo;

struct Material {
    vec3 color;
    vec3 emission;
    float smoothness;
};

layout(set = 1, binding = 0) readonly buffer MaterialBuffer {
    Material list[];
} materials;

struct Circle {
    vec3 position;
    float radius;
    int material;
};

layout(set = 1, binding = 1) readonly buffer CircleBuffer {
    Circle list[];
} circles;

#define MAX_BOUNCE 32
//#define SAMPLES 16

struct HitResult {
    float distance;
    vec3 normal;
    vec3 location;
    int material;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

#define MATERIAL_INVALID -2
#define MATERIAL_FLOOR -1
#define MATERIAL_SKY -3

Material getMaterial(int index) {
    if (index == MATERIAL_FLOOR) {
        Material m;
        m.color = vec3(1.0);
        m.emission = vec3(0.0);
        return m;
    } else if(index == MATERIAL_INVALID) {
        Material m;
        m.color = vec3(1.0, 0.75, 0.79);
        m.emission = vec3(1.0, 0.75, 0.79);
        return m;
    } else if(index == MATERIAL_SKY) {
        vec3 ray_origin = (viewData.worldview * vec4(vec3(0.0), 1.0)).xyz;

        Material m;
        m.color = vec3(0.0);
        m.emission = vec3(0.53, 0.81, 0.92) * (1.0 - ray_origin.y * 0.03);
        return m;
    }
    return materials.list[index];
}

float rand(inout uint state) {
    state = state * 1103515245 + 104723;
    return 1.0 - float(state) / float(uint(-1));
}

vec3 randDirection(inout uint state) {
    float x = rand(state) * 2.0 - 1.0;
    float y = rand(state) * 2.0 - 1.0;
    float z = rand(state) * 2.0 - 1.0;
    return normalize(vec3(x, y, z));
}

vec3 randHemisphere(inout uint state, vec3 normal) {
    vec3 direction = randDirection(state);
    if (dot(direction, normal) < 0.0) {
        direction = -direction;
    }
    return normalize(direction + normal * 2.0);
}

vec3 lerp(vec3 a, vec3 b, float t) {
    return a + (b - a) * t;
}

vec3 getAmbientLight(Ray ray) { // make a background sky color with a sun
    vec3 ray_origin = (viewData.worldview * vec4(vec3(0.0), 1.0)).xyz;
    vec3 ray_direction = normalize(ray.direction);

    float sun_angle = dot(ray_direction, normalize(vec3(0.3, 0.3, 0.3)));
    sun_angle = clamp(sun_angle, 0.0, 1.0);

    vec3 sun_color = vec3(1.0, 0.75, 0.79);
    vec3 sky_color = vec3(0.53, 0.81, 0.92) * (1.0 - ray_origin.y * 0.03);
    return lerp(sky_color, sun_color, sun_angle);
}

bool raySphereIntersect(Ray ray, vec3 spherePosition, float sphereradius, out HitResult result) {
    vec3 oc = ray.origin - spherePosition;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphereradius * sphereradius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant > 0) {
        float t = (-b - sqrt(discriminant)) / (2.0 * a);
        if (t > 0) {
            result.distance = t;
            result.location = ray.origin + ray.direction * t;
            result.normal = normalize(result.location - spherePosition);
            return true;
        }
    }
    return false;
}

bool rayPlaneIntersect(Ray ray, vec3 planePosition, vec3 planeNormal, out HitResult result) {
    float d = -dot(planePosition, planeNormal);
    float v = dot(ray.direction, planeNormal);
    if (abs(v) <= 0) {
        return false;
    }
    float t = -(dot(ray.origin, planeNormal) + d) / v;
    if (t > 0) {
        result.distance = t;
        result.location = ray.origin + ray.direction * t;
        result.normal = planeNormal;
        return true;
    }
    return false;
}


bool rayAllObjects(Ray ray, out HitResult result) {
    bool didHit = false;
    result.distance = 1.0 / 0.0;
    result.normal = vec3(0.0, 0.0, 0.0);

    HitResult r;

    /*// add floor
    if (rayPlaneIntersect(ray, vec3(0.0, -2.0, 0.0), vec3(0.0, 1.0, 0.0), r)) {
        result = r;
        result.material = MATERIAL_FLOOR;
        didHit = true;
    }

    // add sky
    if (rayPlaneIntersect(ray, vec3(0.0, 10.0, 0.0), vec3(0.0, -1.0, 0.0), r) && r.distance < result.distance) {
        result = r;
        result.material = MATERIAL_SKY;
        didHit = true;
    }*/

    for (int i = 0; i < circles.list.length(); i++) {
        Circle circle = circles.list[i];
        if (raySphereIntersect(ray, circle.position, circle.radius, r) && r.distance < result.distance) {
            result = r;
            result.material = circle.material;
            didHit = true;
        }
    }
    return didHit;
}

vec3 rayTrace(Ray ray, inout uint rngState) {
    vec3 color = vec3(1.0);
    vec3 light = vec3(0.0);

    for (int i = 0; i < MAX_BOUNCE; i++) {
        HitResult result;
        if (rayAllObjects(ray, result)) {

            Material m = getMaterial(result.material);

            vec3 diffuseDir = randHemisphere(rngState, result.normal);
            vec3 specularDir = reflect(ray.direction, result.normal);

            light += m.emission * color;
            color *= m.color;

            ray.origin = result.location + result.normal * 0.001;
            ray.direction = lerp(diffuseDir, specularDir, m.smoothness);
        } else {
            light += getAmbientLight(ray) * color;
            break;
        }
    }
    return light;
}

HitResult rayTraceFirstHit(Ray ray, inout uint rngState) {
    HitResult result;
    rayAllObjects(ray, result);
    return result;
}

vec3 rayTraceSampled(Ray ray, inout uint rngState) {
    vec3 light = vec3(0.0);
//    int sample_count = SAMPLES;
    int sample_count = renderInfo.sample_count;
    for(int i = 0; i < sample_count; i++) {

        vec3 jitter = randDirection(rngState) * viewData.blur * 0.01;
        // rotate ray direction
        ray.direction = normalize(ray.direction + jitter);

        light += rayTrace(ray, rngState);
    }
    return light / float(sample_count);
}

uint generateRngSeed() {
//    return uint(gl_FragCoord.y) + uint(gl_FragCoord.x * 1080.0) * uint(renderInfo.time * 10.0);
    return uint(gl_FragCoord.y) + uint(gl_FragCoord.x * 1080.0);
}

vec3 getPixelColor(Ray ray, vec2 coord) {
    uint rngState = generateRngSeed();
    return rayTraceSampled(ray, rngState);
}

void getPixelNormal(Ray ray, vec2 coord, out vec3 albedo, out vec3 normal, out float depth) {
    uint rngState = generateRngSeed();
    HitResult r = rayTraceFirstHit(ray, rngState);
    albedo = getMaterial(r.material).color;
    normal = r.normal;
    depth =  r.distance;
}

void main() {
    mat4 proj = viewData.proj;
    float aspect = proj[0][0] / proj[1][1];

    vec2 real_coord = coord;
    real_coord.x *= aspect;

    vec3 ray_origin = (viewData.worldview * vec4(vec3(0.0), 1.0)).xyz;
    vec3 ray_target = (viewData.worldview * vec4(real_coord, 1.0, 1.0)).xyz;
    vec3 ray_direction = normalize(ray_target - ray_origin);

    Ray ray;
    ray.origin = ray_origin;
    ray.direction = ray_direction.xyz;

    vec3 light = getPixelColor(ray, real_coord);

    vec3 albedo;
    vec3 normal;
    float depth;
    getPixelNormal(ray, real_coord, albedo, normal, depth);

    f_color = vec4(light, 1.0);
    f_albedo = albedo;
    f_normal = normal;
    f_depth = 1.0 / depth;
}
