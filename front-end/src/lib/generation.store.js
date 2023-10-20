import { writable } from "svelte/store";

const API_ENDPOINT = "http://localhost:8504/generate-ticket";

export const generationStates = {
    IDLE: "idle",
    SUCCESS: "success",
    ERROR: "error",
    LOADING: "loading",
};

function createGenerationStore() {
    const { subscribe, update } = writable({ state: generationStates.IDLE, data: { title: "", text: "" } });

    return {
        subscribe,
        generate: async (fromQuestion) => {
            update(() => ({ state: generationStates.LOADING, data: { title: "", text: "" } }));
            try {
                const response = await fetch(`${API_ENDPOINT}?text=${encodeURI(fromQuestion)}`, {
                    method: "GET",
                });
                const generation = await response.json();
                update(() => ({ state: generationStates.SUCCESS, data: generation.result }));
            } catch (e) {
                console.log("e: ", e);
                update(() => ({ state: generationStates.ERROR, data: { title: "", text: "" } }));
            }
        },
    };
}

export const generationStore = createGenerationStore();
