<script>
    import { createEventDispatcher, onMount } from "svelte";
    import { generationStates, generationStore } from "./generation.store";

    /** @type {HTMLDialogElement | undefined}*/
    let modal;

    const dispatch = createEventDispatcher();

    onMount(() => {
        modal.showModal();
        modal.addEventListener("close", onClose);
        return () => modal.removeEventListener("close", onClose);
    });

    function onClose() {
        dispatch("close");
    }
</script>

<dialog
    bind:this={modal}
    class="inset-0 w-full md:w-1/2 h-1/2 p-4 rounded-lg border border-indigo-200 shadow-lg relative"
>
    <form class="flex flex-col justify-between h-full" method="dialog">
        <div class="flex flex-col">
            <h1 class="text-2xl">Create new internal ticket</h1>
            <div>
                <label class="block pl-2"
                    >Title <br />
                    <input type="text" class="border w-full text-lg px-2" value={$generationStore.data.title} /></label
                >
            </div>
            <div class="mt-8">
                <label class="block pl-2"
                    >Body <br />
                    <textarea class="border rounded-sm w-full h-64 p-2" value={$generationStore.data.text} /></label
                >
            </div>
        </div>
        <button type="submit" class="bg-indigo-500 text-white rounded-lg px-4 py-2">Submit</button>
    </form>
    {#if $generationStore.state === generationStates.LOADING}
        <div class="absolute inset-0 bg-indigo-100 bg-opacity-90 flex justify-center items-center">
            Generating title and question body...
        </div>
    {/if}
    <div class="absolute top-0 right-2 text-gray-300 hover:text-gray-900">
        <button class="text-2xl" on:click={onClose}>×</button>
    </div>
</dialog>

<style>
    dialog::backdrop {
        @apply bg-gradient-to-t from-white to-indigo-500;
        opacity: 0.75;
    }
</style>
